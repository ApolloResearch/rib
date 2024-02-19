import json
import time
from typing import Callable, Literal, Optional, Union

import numpy as np
import torch
from jaxtyping import Bool, Float
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from rib.data import (
    HFDatasetConfig,
    ModularArithmeticDatasetConfig,
    VisionDatasetConfig,
)
from rib.data_accumulator import Edges
from rib.hook_fns import edge_ablation_forward_hook_fn, rotate_pre_forward_hook_fn
from rib.hook_manager import Hook, HookedModel
from rib.linalg import calc_rotation_matrix
from rib.loader import load_model_and_dataset_from_rib_config
from rib.log import logger
from rib.models import MLP, SequentialTransformer
from rib.rib_builder import RibBuildResults
from rib.settings import REPO_ROOT
from rib.types import TORCH_DTYPES, RootPath, StrDtype
from rib.utils import (
    check_out_file_overwrite,
    eval_cross_entropy_loss,
    eval_model_accuracy,
    load_config,
    set_seed,
)

BasisVecs = Union[Float[Tensor, "orig rib"], Float[Tensor, "orig orig"]]
BasisVecsPinv = Union[Float[Tensor, "rib orig"], Float[Tensor, "orig orig"]]
AblationAccuracies = dict[str, dict[int, float]]
EdgeMasks = dict[str, dict[int, Bool[Tensor, "rib_out rib_in"]]]


class StaticScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schedule_type: Literal["exponential", "linear"] = Field(
        ...,
        description="The type of ablation schedule to use. 'exponential' uses an exponential "
        "schedule, 'linear' uses a linear schedule.",
    )
    early_stopping_threshold: Optional[float] = Field(
        None,
        description="The threshold to use for stopping the ablation calculations early. The"
        "experiment will stop when the ablated score is more than `early_stopping_threshold` away "
        "from the unablated score. If None, we don't stop early.",
    )
    specific_points: Optional[list[int]] = Field(
        None,
        description="A list of number of vecs remaining to add to the schedule. If None, we use"
        "the default schedule.",
    )


class StaticSchedule:
    def __init__(self, config: StaticScheduleConfig, n_vecs: int):
        self.config: StaticScheduleConfig = config
        self._n_vecs: int = n_vecs
        # Early stopping variables
        self._base_score: Optional[float] = None
        self._stop_iteration: bool = False
        # Generate the ablation schedule
        initial_ablation_schedule = self._get_initial_ablation_schedule()
        # Add specific points that were manually requested
        self._ablation_schedule: list[int] = self._add_specific_ablation_points(
            initial_ablation_schedule
        )

    def _add_specific_ablation_points(self, ablation_schedule: list[int]) -> list[int]:
        """Add each number of vecs remaining in self.specific_points to the ablation schedule."""
        if self.config.specific_points is not None:
            # Ignore the specific points that are greater than the number of vecs
            specific_ablated_vecs = [
                self._n_vecs - x for x in self.config.specific_points if x <= self._n_vecs
            ]
            # Add our specific points for the number of vecs remaining to the ablation schedule
            ablation_schedule = sorted(
                list(set(ablation_schedule + specific_ablated_vecs)), reverse=True
            )
        return ablation_schedule

    def _check_early_stopping(self, score: float, early_stopping_threshold: float) -> bool:
        """Check if we should stop early.

        Stop if the score is more than `early_stopping_threshold` away from the base (no ablations)
        result, i.e. once we ablated to many vecs that the model output is crap, don't bother
        ablating even more vecs.

        Note that this functionality assumes that the first call to this method is the base result,
        and that later steps are ablating more vecs than earlier steps.
        """
        # This assumes that the first call is the base result (n_vecs_ablated = 0)
        if self._base_score is None:
            self._base_score = score
        # Stop if the score is more than `early_stopping_threshold` away from the base result.
        if abs(score - self._base_score) > early_stopping_threshold:
            logger.info(f"Stopping early with {score=}, {self._base_score=} ")
            return True
        else:
            return False

    def _get_initial_ablation_schedule(self) -> list[int]:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def size(self):
        return len(self._ablation_schedule)

    def __iter__(self):
        # For early stopping we need to iterate from low to high n_vecs_ablated, thus [::-1].
        for n_vecs_remaining in self._ablation_schedule[::-1]:
            if self._stop_iteration:
                break
            else:
                yield n_vecs_remaining

    def update_early_stopping_flag(self, score: float):
        if self.config.early_stopping_threshold is not None:
            self._stop_iteration = self._check_early_stopping(
                score, self.config.early_stopping_threshold
            )


class LinearScheduleConfig(StaticScheduleConfig):
    schedule_type: Literal["linear"]
    n_points: int = Field(
        ...,
        description=(
            "The number of points to use in the linear ablation schedule. Must be specified if "
            "schedule_type is linear and cannot be specified if schedule_type is exponential."
        ),
    )


class LinearSchedule(StaticSchedule):
    config: LinearScheduleConfig

    def _get_initial_ablation_schedule(self) -> list[int]:
        """Create a linear schedule for the number of vectors to ablate.

        The points are evenly spaced between `n_vecs` and 0, including the endpoints and any points
        in `self.specific_points` are also added.

        Returns:
            An iterable schedule for the number of vectors to ablate.

        Examples:
            n_points = 3, n_vecs = 12 --> [12, 6, 0]
            n_points = 11, n_vecs = 120 --> [10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0]

        """
        assert self.config.n_points >= 2, f"{self.config.n_points} must be at least 2."
        assert (
            self.config.n_points <= self._n_vecs + 1
        ), f"{self.config.n_points} must be <= {self._n_vecs+1}."

        ablation_schedule = [int(a) for a in np.linspace(self._n_vecs, 0, self.config.n_points)]

        return ablation_schedule


class ExponentialScheduleConfig(StaticScheduleConfig):
    schedule_type: Literal["exponential"]
    ablate_every_vec_cutoff: Optional[int] = Field(
        None,
        description="The point in the exponential schedule at which we start ablating every"
        "individual vector. If None, always ablate every vector.",
    )
    exp_base: float = Field(2.0, description="The base of the exponential schedule.")


class ExponentialSchedule(StaticSchedule):
    config: ExponentialScheduleConfig

    def _get_initial_ablation_schedule(self) -> list[int]:
        """Create an exponential schedule for the number of vectors to ablate.

        The schedule is exponential with a base of 2, with the exceptions
        * we test all values of n_vecs_remaining from 0 to `self.ablate_every_vec_cutoff`
        * we test ablating no vectors (n_vecs_remaining = n_vecs)
        * we test manually specified points given in `config.specific_points`.

        Returns:
            The schedule for the number of vectors to ablate.

        Examples (all with exp_base = 2):
            ablate_every_vec_cutoff = None, n_vecs = 12 --> [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
            0]
            ablate_every_vec_cutoff = 0, n_vecs = 12 --> [12, 11, 9, 5, 0]
            ablate_every_vec_cutoff = 1, n_vecs = 12 --> [12, 11, 10, 8, 4, 0]
            ablate_every_vec_cutoff = 3, n_vecs = 12 --> [12, 11, 10, 9, 8, 6, 2, 0]
            ablate_every_vec_cutoff = 3, n_vecs = 24 --> [24, 23, 22, 21, 20, 18, 14, 6, 0]
        """
        cutoff = self.config.ablate_every_vec_cutoff
        exp_base = self.config.exp_base

        if cutoff is None:
            return list(range(self._n_vecs, -1, -1))

        assert cutoff < self._n_vecs, "ablate_every_vec_cutoff must be smaller than n_vecs"
        assert cutoff >= 0, "ablate_every_vec_cutoff must be positive"
        # The section in which we ablate every vector.
        ablate_every_vecs: list[int] = list(range(self._n_vecs, self._n_vecs - cutoff - 1, -1))
        # The section in which we ablate according to 2^x.
        ablate_exponential: list[int] = []
        prev_val = ablate_every_vecs[-1]
        for x in range(self._n_vecs):
            exp_val = int(prev_val - exp_base**x)
            if exp_val > 0:
                ablate_exponential.append(exp_val)
                prev_val = exp_val
            else:
                # No more values to append, just add the case for no ablation and exit
                ablate_exponential.append(0)
                break

        # combine the two sections
        ablation_schedule = ablate_every_vecs + ablate_exponential
        assert (
            ablation_schedule[0] == self._n_vecs
        ), "The first element of the schedule must be n_vecs."
        assert ablation_schedule[-1] == 0, "The last element of the schedule must be 0."

        return ablation_schedule


class BisectScheduleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schedule_type: Literal["bisect"] = Field(
        "bisect",
        description="The type of ablation schedule to use. 'bisect' uses a bisect schedule.",
    )
    score_target: float = Field(
        ...,
        description="The target loss value for the bisect schedule.",
    )
    scaling: Literal["linear", "logarithmic"] = Field(
        "linear",
        description="Whether to use the linear or logarithmic midpoint for bisect.",
    )


class BisectSchedule:
    """Implement a bisect search to find the largest n_vec_ablated given target_score.

    Find the largest n_vec_ablated (smallest n_vec_remaining) such that the loss is equal or less
    than the target_loss. "Find" here means that, after the run, self._upper_bound will be the
    largest n_vec_ablated where the loss is <= the target_loss, and self._lower_bound will be the
    smallest n_vec_ablated where the loss is > the target_loss.
    Whether an accuracy score or a loss-type score should be assumed is read frome eval_type
    in the ablation config (passed via this classes init).

    Examples:
        n_eigenvecs = 20, target loss = -3, linear scaling
            First try: n_ablated_vecs = 10. Loss bad (-2.9)
                Set upper bound to 10 (can ablate at most 10 vecs)
            Second try: n_ablated_vecs = 15. Loss good (-3.3)
                Set lower bound to 15 (can't ablate more than 15 vecs)
            Third try: n_ablated_vecs = 12. Loss good (-3.1)
                Set lower bound to 12 (can't ablate more than 12 vecs)
            Fourth try: n_ablated_vecs = 11. Loss bad (-2.98)
                Set upper bound to 11 (can ablate at most 11 vecs)
            Exit with upper bound = 11, lower bound = 12.
                We can ablate 11 vecs, and ablating 12 vecs is too much.
    """

    def __init__(
        self,
        config: BisectScheduleConfig,
        n_vecs: int,
        eval_type: Literal["accuracy", "ce_loss"],
    ):
        self.config: BisectScheduleConfig = config
        self._eval_type: Literal["accuracy", "ce_loss"] = eval_type
        self._upper_bound: int = n_vecs
        assert self._upper_bound > 0, "n_vecs must be positive"
        self._lower_bound: int = 0
        self._most_recent_proposal: int = -1

    def _get_proposal(self) -> int:
        if self.config.scaling == "linear":
            proposal = (self._upper_bound + self._lower_bound) // 2
        elif self.config.scaling == "logarithmic":
            proposal = int(np.exp((np.log(self._upper_bound) + np.log(self._lower_bound)) / 2))
        else:
            raise ValueError(f"Invalid scaling: {self.config.scaling}")
        # Avoid getting stuck due to rounding
        if proposal == self._lower_bound:
            proposal += 1
        if proposal == self._upper_bound:
            proposal -= 1
        # Check that we don't forget .update_bounds() somewhere
        if proposal == self._most_recent_proposal:
            raise RuntimeError("Ablation schedule stuck. Did you call .update_bounds(score)?")
        self._most_recent_proposal = proposal
        return proposal

    def size(self):
        return None

    def __iter__(self):
        while self._upper_bound - self._lower_bound > 1:
            yield self._get_proposal()

    def update_bounds(self, score: float):
        """Bisect logic: update either the upper or lower bound based on the current score."""
        # Loss: Lower is better. Check if the current loss is good, i.e. <= the target loss.
        if self._eval_type == "ce_loss":
            if score <= self.config.score_target:
                # Good loss --> ablate more vectors. Set lower bound to current n_vecs_ablated.
                self._lower_bound = self._most_recent_proposal
            else:
                # Bad loss --> ablate fewer vectors. Set upper bound to current n_vecs_ablated.
                self._upper_bound = self._most_recent_proposal
        # The same as above but opposite if statements because higher accuracy is better.
        elif self._eval_type == "accuracy":
            if score >= self.config.score_target:
                self._lower_bound = self._most_recent_proposal
            else:
                self._upper_bound = self._most_recent_proposal
        else:
            raise ValueError(f"Invalid score_type: {self.config.score_type}")


ScheduleConfig = Union[LinearScheduleConfig, ExponentialScheduleConfig, BisectScheduleConfig]


class AblationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    exp_name: str
    out_dir: Optional[RootPath] = Field(
        REPO_ROOT / "rib_scripts/ablations/out",
        description="Directory for the output files. Defaults to `./out/`. If None, no output "
        "is written. If a relative path, it is relative to the root of the rib repo.",
    )
    ablation_type: Literal["rib", "orthogonal", "edge"] = Field(
        description="The type of ablation to perform. 'rib' ablates nodes from the RIB basis (C)."
        "'orthogonal' ablates nodes from the svd/pca basis (W, or YU). Edge ablation ablates edges"
        "connecting RIB nodes. It uses the C matrices as we don't have edges for W."
    )
    rib_results_path: RootPath
    schedule: ScheduleConfig = Field(
        ...,
        discriminator="schedule_type",
        description="The schedule to use for ablations.",
    )
    dataset: Union[ModularArithmeticDatasetConfig, HFDatasetConfig, VisionDatasetConfig] = Field(
        ...,
        discriminator="dataset_type",
        description="The dataset to use to build the graph.",
    )
    ablation_node_layers: list[str]
    batch_size: int
    dtype: StrDtype
    seed: int
    eval_type: Literal["accuracy", "ce_loss"] = Field(
        ...,
        description="The type of evaluation to perform on the model before building the graph.",
    )


def _get_schedule_from_config(
    schedule_config: ScheduleConfig, n_vecs: int, eval_type: Literal["accuracy", "ce_loss"]
) -> Union[ExponentialSchedule, LinearSchedule, BisectSchedule]:
    if schedule_config.schedule_type == "linear":
        return LinearSchedule(schedule_config, n_vecs)
    elif schedule_config.schedule_type == "exponential":
        return ExponentialSchedule(schedule_config, n_vecs)
    elif schedule_config.schedule_type == "bisect":
        return BisectSchedule(schedule_config, n_vecs, eval_type)


@torch.inference_mode()
def ablate_node_layers_and_eval(
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]],
    ablation_node_layers: list[str],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    eval_fn: Callable,
    eval_type: Literal["accuracy", "ce_loss"],
    module_names: list[str],
    schedule_config: ScheduleConfig,
    device: str,
    dtype: Optional[torch.dtype] = None,
) -> AblationAccuracies:
    """Rotate to and from a truncated basis and compare ablation accuracies/losses.

    Note that we want our ablation schedules for different bases to match up, even though different
    bases may have different number of basis vectors due to truncation. We therefore create our
    ablation schedule assuming a non-truncated basis (i.e. using the `orig` size
    (basis_vecs.shape[0])).

    Args:
        basis_matrices: List of basis vector matrices and their pseudoinverses. In the orthogonal
            basis case, the pseudoinverse is the transpose.
        ablation_node_layers: The names of the node layers whose (rotated) inputs we want to ablate.
        hooked_model: The hooked model.
        data_loader: The data loader to use for testing.
        eval_fn: The function to use to evaluate the model.
        module_names: The names of the modules to apply the ablations. Can be any valid pytorch
            module in hooked_model.model. These typically correspond to section_names (e.g.
            "sections.section_0") when the model is a SequentialTransformer or raw layers (e.g.
            "layers.2") when the model is an MLP.
        schedule_config: The config for the ablation schedule.
        device: The device to run the model on.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.

    Returns:
        A dictionary mapping node layers to ablation accuracies/losses.
    """
    results: AblationAccuracies = {}
    for ablation_node_layer, module_name, (basis_vecs, basis_vecs_pinv) in zip(
        ablation_node_layers, module_names, basis_matrices, strict=True
    ):
        ablation_schedule = _get_schedule_from_config(
            schedule_config, basis_vecs.shape[0], eval_type
        )
        base_score: Optional[float] = None

        # Track the results for the case when there is no ablation. There may be many of these, so we
        # store them to avoid recomputing.
        n_truncated_vecs = basis_vecs.shape[0] - basis_vecs.shape[1]

        results[ablation_node_layer] = {}
        # Iterate through possible number of ablated vectors, starting from no ablated vectors
        for n_ablated_vecs in tqdm(
            ablation_schedule,
            total=ablation_schedule.size(),
            desc=f"Ablating {module_name}",
        ):
            # Note that we may have truncated vectors with small eigenvalues, so we make sure not to
            # ablate more vectors than we have remaining
            n_vecs_remaining = basis_vecs.shape[0] - n_ablated_vecs

            # Count the n_ablated_vecs taking into account the truncation
            n_ablated_vecs_trunc = max(n_ablated_vecs - n_truncated_vecs, 0)

            if n_ablated_vecs_trunc == 0 and base_score is not None:
                # We may have already calculated the score for the case when there is no ablation
                results[ablation_node_layer][n_vecs_remaining] = base_score
                continue

            basis_vecs = basis_vecs.to(device)
            basis_vecs_pinv = basis_vecs_pinv.to(device)
            rotation_matrix = calc_rotation_matrix(
                vecs=basis_vecs,
                vecs_pinv=basis_vecs_pinv,
                n_ablated_vecs=n_ablated_vecs_trunc,
            )

            rotation_hook = Hook(
                name=ablation_node_layer,
                data_key="rotation",
                fn=rotate_pre_forward_hook_fn,
                module_name=module_name,
                fn_kwargs={"rotation_matrix": rotation_matrix},
            )

            score = eval_fn(
                hooked_model, data_loader, hooks=[rotation_hook], dtype=dtype, device=device
            )
            results[ablation_node_layer][n_vecs_remaining] = score

            if isinstance(ablation_schedule, BisectSchedule):
                ablation_schedule.update_bounds(score)
            else:
                ablation_schedule.update_early_stopping_flag(score)

        # Sort the results by the number of edges kept, descending
        results[ablation_node_layer] = dict(
            sorted(results[ablation_node_layer].items(), reverse=True)
        )

    return results


def _get_edge_mask(
    edge_weights: Float[Tensor, "rib_out rib_in"], num_edges_kept: int, keep_const_edges: bool
) -> Bool[Tensor, "rib_out rib_in"]:
    """
    Returns a mask over edge weights, which keeps the largest edges.

    Args:
        edge_weights: A tensor representing edge weights.
        num_edges_kept: The number of edges to keep. If this number is greater than the total
            number of edges, all edges are kept.
        keep_const_edges: A flag to indicate if we should keep all edges in the first col
            (representing edges that originate at the constant node). These edges are 'free', in that they don't count towards num_edges_kept.

    Returns:
        Bool tensor of the same shape as the edge edges.

    Example:
        >>> edge_weights = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        >>> _get_edge_mask(edge_weights, 2, True)
        tensor([[True, False, False],
                [True, False, False],
                [True, True,  True]])
    """
    sub_weights = edge_weights[:, 1:] if keep_const_edges else edge_weights
    if num_edges_kept > sub_weights.numel():  # keep all edges
        return torch.ones_like(edge_weights, dtype=torch.bool)
    if num_edges_kept == 0:  # ablate no edges
        sub_mask = torch.zeros_like(sub_weights, dtype=torch.bool)
    else:  # keep some edges
        threshold = torch.topk(sub_weights.flatten(), k=num_edges_kept).values[-1]
        sub_mask = sub_weights >= threshold
    # transform sub_mask back to full size
    if keep_const_edges:
        full_mask = torch.ones_like(edge_weights, dtype=torch.bool)
        full_mask[:, 1:] = sub_mask
    else:
        full_mask = sub_mask
    return full_mask


@torch.inference_mode()
def ablate_edges_and_eval(
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]],
    ablation_node_layers: list[str],
    edges: list[Edges],
    hooked_model: HookedModel,
    data_loader: DataLoader,
    eval_fn: Callable,
    eval_type: Literal["accuracy", "ce_loss"],
    module_names: list[str],
    schedule_config: ScheduleConfig,
    device: str,
    dtype: Optional[torch.dtype] = None,
    always_keep_const_dir=False,
) -> tuple[AblationAccuracies, EdgeMasks, dict[str, int]]:
    """Perform a series of edge ablation experiments across layers and multiple # of edges to keep.

    Note that we want our ablation schedules for different bases to match up, even though different
    bases may have different number of basis vectors due to truncation. We therefore create our
    ablation schedule assuming a non-truncated basis (i.e. using the full possible # of edges,
    out_dim * in_dim. This can be very large).

    Args:
        basis_matrices: List of basis vector matrices (Cs) and their pseudoinverses (C_pinv).
        ablation_node_layers: The names of the node layers whose (rotated) inputs we want to ablate.
        edges: The edge weights computed by RIB. We use these to determine edge ablation order.
        hooked_model: The hooked model.
        data_loader: The data loader to use for testing.
        eval_fn: The function to use to evaluate the model.
        module_names: The names of the modules to apply the ablations. Can be any valid pytorch
            module in hooked_model.model. These typically correspond to section_names (e.g.
            "sections.section_0") when the model is a SequentialTransformer or raw layers (e.g.
            "layers.2") when the model is an MLP.
        schedule_config: The config for the ablation schedule.
        device: The device to run the model on.
        dtype: The data type to cast the inputs to. Ignored if int32 or int64.
        keep_const_edges: Used to always keep the constant edges (for free) when ablating a
            centered RIB graph.

    Returns:
        A dictionary mapping node layers to ablation accuracies/losses.
        A dictionary mapping node layers to edge masks.
        A dictionary mapping node layers to the number of edges required to achieve the target
            accuracy/loss (for bisect schedule only, otherwise empty dict).
    """
    base_score = eval_fn(hooked_model, data_loader, hooks=[], dtype=dtype, device=device)

    results: AblationAccuracies = {}
    edge_masks: EdgeMasks = {}
    n_edges_required = {}
    basis_pairs = zip(basis_matrices[:-1], basis_matrices[1:])
    for ablation_node_layer, module_name, basis_pair, layer_edges in zip(
        ablation_node_layers[:-1], module_names[:-1], basis_pairs, edges, strict=True
    ):
        (in_C, in_C_inv), (out_C, out_C_inv) = basis_pair
        total_possible_edges = in_C.shape[0] * out_C.shape[0]

        ablation_schedule = _get_schedule_from_config(
            schedule_config, total_possible_edges, eval_type
        )

        results[ablation_node_layer] = {}
        edge_masks[ablation_node_layer] = {}
        # Iterate through possible number of ablated edges, starting from no ablated edges
        for num_edges_ablated in tqdm(
            ablation_schedule, total=ablation_schedule.size(), desc=f"Ablating {module_name}"
        ):
            num_edges_kept = total_possible_edges - num_edges_ablated
            edge_mask = _get_edge_mask(
                edge_weights=layer_edges.E_hat,
                num_edges_kept=num_edges_kept,
                keep_const_edges=always_keep_const_dir,
            )
            if edge_mask.all():
                score = base_score
            else:
                edge_masks[ablation_node_layer][num_edges_kept] = edge_mask
                hook = Hook(
                    name=module_name,
                    data_key="edge_ablation",
                    fn=edge_ablation_forward_hook_fn,
                    module_name=module_name,
                    fn_kwargs={
                        "edge_mask": edge_mask.to(device),
                        "in_C": in_C.to(device),
                        "in_C_inv": in_C_inv.to(device),
                        "out_C": out_C.to(device),
                        "out_C_inv": out_C_inv.to(device),
                    },
                )
                score = eval_fn(hooked_model, data_loader, hooks=[hook], dtype=dtype, device=device)

            results[ablation_node_layer][num_edges_kept] = score

            if isinstance(ablation_schedule, BisectSchedule):
                ablation_schedule.update_bounds(score)
            else:
                ablation_schedule.update_early_stopping_flag(score)

        # Sort the results by the number of edges kept, descending
        results[ablation_node_layer] = dict(
            sorted(results[ablation_node_layer].items(), reverse=True)
        )

        if isinstance(ablation_schedule, BisectSchedule):
            n_edges_required[ablation_node_layer] = (
                total_possible_edges - ablation_schedule._upper_bound
            )

    return results, edge_masks, n_edges_required


def load_basis_matrices(
    rib_results: RibBuildResults,
    ablation_node_layers: list[str],
    ablation_type: Literal["rib", "orthogonal", "edge"],
    dtype: torch.dtype,
    device: str,
) -> list[tuple[BasisVecs, BasisVecsPinv]]:
    """Load the basis matrices and their pseudoinverses.

    Uses C and C_pinv for 'rib' ablations and W and W_pinv for 'orthogonal' ablations.

    If ablation type is 'edge' will return idenity matricies in place of None matricies. Otherwise
    assert all matricies are non-None.

    Args:
        rib_results: The results of building the RIB graph.
        ablation_node_layers: The node layers to ablate.
        ablation_type: The type of ablation to perform ('rib', 'orthogonal', 'edge').
        dtype: The data type to cast the basis matrices to.
        device: The device to load the basis matrices to.

    Returns:
        - A list of basis matrices.
        - A list of pseudoinverse basis matrices.
    """

    # Get the basis vecs and their pseudoinverses using the module_names as keys
    basis_matrices: list[tuple[BasisVecs, BasisVecsPinv]] = []
    basis_infos_dict = {info.node_layer: info for info in rib_results.interaction_rotations}

    for module_name in ablation_node_layers:
        basis_info = basis_infos_dict[module_name]
        if ablation_type == "rib":
            assert basis_info.C is not None, f"{module_name} has no C matrix."
            assert basis_info.C_pinv is not None, f"{module_name} has no C_pinv matrix."
            basis_vecs = basis_info.C.to(dtype=dtype, device=device)
            basis_vecs_pinv = basis_info.C_pinv.to(dtype=dtype, device=device)
        elif ablation_type == "orthogonal":
            assert basis_info.W is not None, f"{module_name} has no W matrix."
            assert basis_info.W_pinv is not None, f"{module_name} has no W_pinv matrix."
            basis_vecs = basis_info.W.to(dtype=dtype, device=device)
            basis_vecs_pinv = basis_info.W_pinv.to(dtype=dtype, device=device)
        else:
            assert ablation_type == "edge"
            if basis_info.C is None:
                assert basis_info.C_pinv is None
                basis_vecs = torch.eye(basis_info.orig_dim, dtype=dtype, device=device)
                basis_vecs_pinv = torch.eye(basis_info.orig_dim, dtype=dtype, device=device)
            else:
                assert basis_info.C_pinv is not None
                basis_vecs = basis_info.C.to(dtype=dtype, device=device)
                basis_vecs_pinv = basis_info.C_pinv.to(dtype=dtype, device=device)
        basis_matrices.append((basis_vecs, basis_vecs_pinv))
    return basis_matrices


def load_bases_and_ablate(
    config_path_or_obj: Union[str, AblationConfig], force: bool = False
) -> dict:
    """Load basis matrices and run ablation experiments.

    The process is as follows:
        1. Load pre-saved basis matrices (typcially RIB bases (Cs) or orthogonal bases (Ws)).
        2. Load the corresponding model and dataset (the dataset may be non-overlapping with
            that used to create the basis matrices).
        3. For each number of ablated nodes `n`, create a rotation matrix that has the effect of
        rotating to and from the new basis with `n` fewer basis vectors.
        4. Run the test set through the model, applying the above rotations at each node layer, and
        calculate the resulting accuracy/loss.
        5. Repeat steps 3 and 4 for a range of values of n, storing the resulting accuracies/losses.

    Note that we don't create a node layer at the output of the final module, as ablating nodes in
    this layer is not useful.

    Args:
        config_path_or_obj: The path to the config file or the config object itself.
        force: Whether to overwrite existing output files.

    Returns:
        A dictionary containing the results of the ablation experiments. The dictionary contains
        "config" (json dump of the config), "results" (a dictionary mapping node layers to
        accuracies/losses), "time_taken" (the time taken to run the ablations), and
        "no_ablation_result" (numbering). If "edge" ablations were performed, the dictionary also
        contains "n_edges_required" (the number of edges required to achieve the target accuracy/
        loss, non-empty only if BisectSchedule was used) and "edge_masks" (a dictionary mapping
        node layers to edge masks).
        If the config has an out_dir, the results are also written to a file in that directory.
    """
    start_time = time.time()
    config = load_config(config_path_or_obj, config_model=AblationConfig)

    if config.out_dir is not None:
        config.out_dir.mkdir(parents=True, exist_ok=True)
        out_file = (
            config.out_dir / f"{config.exp_name}_{config.ablation_type}_ablation_results.json"
        )
        if not check_out_file_overwrite(out_file, force):
            raise FileExistsError("Not overwriting output file")

    set_seed(config.seed)
    rib_results = RibBuildResults(**torch.load(config.rib_results_path))

    assert set(config.ablation_node_layers) <= set(
        rib_results.config.node_layers
    ), "The node layers in the config must be a subset of the node layers in the RIB graph."
    if config.ablation_type == "edge":
        # config.node_layers must be a subsequence of loaded_config.node_layers
        assert "|".join(config.ablation_node_layers) in "|".join(
            rib_results.config.node_layers
        ), "node_layers in the config must be a subsequence of the node layers in the RIB graph."
        assert len(rib_results.edges) > 0, "No edges found in the RIB results."
        assert rib_results.contains_all_edges
    else:
        assert "output" not in config.ablation_node_layers, "Cannot ablate the output node layer."

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = TORCH_DTYPES[config.dtype]

    basis_matrices = load_basis_matrices(
        rib_results=rib_results,
        ablation_node_layers=config.ablation_node_layers,
        ablation_type=config.ablation_type,
        dtype=dtype,
        device=device,
    )

    model, dataset = load_model_and_dataset_from_rib_config(
        rib_results.config,
        dataset_config=config.dataset,
        device=device,
        dtype=dtype,
        node_layers=config.ablation_node_layers,
    )
    model.to(device=torch.device(device), dtype=dtype)
    data_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False)
    hooked_model = HookedModel(model)

    # Test model accuracy/loss before running ablations, ta be sure
    eval_fn: Callable = (
        eval_model_accuracy if config.eval_type == "accuracy" else eval_cross_entropy_loss
    )
    no_ablation_result = eval_fn(hooked_model, data_loader, dtype=dtype, device=device)
    logger.info("Model %s on dataset: %.4f", config.eval_type, no_ablation_result)

    if isinstance(model, MLP):
        module_names = config.ablation_node_layers
    else:
        assert isinstance(model, SequentialTransformer)
        module_names = [f"sections.{sec}" for sec in model.sections if sec != "pre"]

    if config.ablation_type == "edge":
        edges_dict = {info.in_node_layer: info for info in rib_results.edges}
        edges = [edges_dict[layer] for layer in config.ablation_node_layers[:-1]]
        ablation_results, edge_masks, n_edges_required = ablate_edges_and_eval(
            basis_matrices=basis_matrices,
            ablation_node_layers=config.ablation_node_layers,
            edges=edges,
            hooked_model=hooked_model,
            data_loader=data_loader,
            eval_fn=eval_fn,
            eval_type=config.eval_type,
            module_names=module_names,
            schedule_config=config.schedule,
            device=device,
            dtype=dtype,
            always_keep_const_dir=rib_results.config.center,
        )
    else:
        ablation_results = ablate_node_layers_and_eval(
            basis_matrices=basis_matrices,
            ablation_node_layers=config.ablation_node_layers,
            hooked_model=hooked_model,
            data_loader=data_loader,
            eval_fn=eval_fn,
            eval_type=config.eval_type,
            module_names=module_names,
            schedule_config=config.schedule,
            device=device,
            dtype=dtype,
        )

    time_taken = f"{(time.time() - start_time) / 60:.1f} minutes"
    logger.info("Finished in %s.", time_taken)

    results = {
        "config": json.loads(config.model_dump_json()),
        "results": ablation_results,
        "time_taken": time_taken,
        "no_ablation_result": no_ablation_result,
    }
    if config.ablation_type == "edge":
        results["n_edges_required"] = n_edges_required
        results["edge_masks"] = edge_masks

    if config.out_dir is not None:
        with open(out_file, "w") as f:
            json.dump(results, f, default=lambda x: x.tolist(), indent=1)
        logger.info("Wrote results to %s", out_file)

    return results

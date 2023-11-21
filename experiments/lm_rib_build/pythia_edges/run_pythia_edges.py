import subprocess
from typing import Any, Dict

import yaml
from torch.cuda import device_count

from rib.log import logger

with open("experiments/lm_rib_build/pythia-14m.yaml") as f:
    base: Dict[str, Any] = yaml.safe_load(f)

node_layers = base["node_layers"]

for i in range(len(node_layers) - 1):
    edge_config = base.copy()
    edge_config["calculate_edges"] = True
    edge_config[
        "interaction_matrices_path"
    ] = "/mnt/ssd-apollo/nix/rib/experiments/lm_rib_build/out/pythia-14m_rib_Cs.pt"
    edge_config["node_layers"] = [node_layers[i], node_layers[i + 1]]
    edge_config["calculate_edges"] = True
    edge_config["exp_name"] = f"edges_{i}"
    edge_config["out_dir"] = f"experiments/lm_rib_build/pythia_edges/out/"

    with open(f"experiments/lm_rib_build/pythia_edges/specs/edges_{i}.yaml", "w") as f:
        yaml.dump(edge_config, f)


NUM_PROCESSES = device_count()
for i in range(len(node_layers) - 1):
    logger.info(f"**** EDGE RUN {i}/{len(node_layers) - 1} ****")
    run_path = "experiments/lm_rib_build/run_lm_rib_build.py"
    spec_path = f"experiments/lm_rib_build/pythia_edges/specs/edges_{i}.yaml"
    subprocess.run(
        ["mpirun", "-n", str(NUM_PROCESSES), "python", run_path, spec_path, "--force"],
        check=True,
    )

import os
from pathlib import Path

# Huggingface cache
os.environ["HF_HOME"] = "/mnt/ssd-interp/huggingface_cache/"

REPO_ROOT = (
    Path(os.environ["GITHUB_WORKSPACE"]) if os.environ.get("CI") else Path(__file__).parent.parent
)

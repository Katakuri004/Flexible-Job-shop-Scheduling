"""Patch first code cell of marl_training.ipynb with instance catalog."""
import json
from pathlib import Path

nb_path = Path(__file__).resolve().parent.parent / "notebooks" / "marl_training.ipynb"
nb = json.loads(nb_path.read_text(encoding="utf-8"))

src = nb["cells"][1]["source"]
text = "".join(src) if isinstance(src, list) else src

old_block = (
    "# Use project_root so this works whether you run from repo root or notebooks/\n\n\n"
    "instance_path = project_root / \"benchmarks\" / \"flexible jobshop\" / \"instances\" / \"Brandimarte1993\" / \"mk01.txt\"\n\n"
    "train_config = TrainingConfig(\n"
    "    instance_path=instance_path,\n"
    "    seed_config=SeedConfig(base_seed=2026),\n"
    "    num_episodes=100,\n"
    "    max_steps_per_episode=64,\n"
    "    log_interval=20,\n"
    "    device=\"cuda\",\n"
    ")\n"
    "train_config"
)

new_block = (
    "# Instance catalog and selection\n"
    "INSTANCE_CATALOG = {\n"
    "    \"toy\": {\"path\": project_root / \"data\" / \"brandimarte_mk_toy.txt\", \"max_steps\": 64, \"bks\": None},\n"
    "    \"mk01\": {\"path\": project_root / \"benchmarks\" / \"flexible jobshop\" / \"instances\" / \"Brandimarte1993\" / \"mk01.txt\", \"max_steps\": 64, \"bks\": 40},\n"
    "}\n"
    "instance_name = \"mk01\"\n"
    "instance_path = INSTANCE_CATALOG[instance_name][\"path\"]\n"
    "instance_max_steps = INSTANCE_CATALOG[instance_name][\"max_steps\"]\n"
    "instance_bks = INSTANCE_CATALOG[instance_name][\"bks\"]\n\n"
    "train_config = TrainingConfig(\n"
    "    instance_path=instance_path,\n"
    "    seed_config=SeedConfig(base_seed=2026),\n"
    "    num_episodes=100,\n"
    "    max_steps_per_episode=instance_max_steps,\n"
    "    log_interval=20,\n"
    "    device=\"cuda\" if torch.cuda.is_available() else \"cpu\",\n"
    ")\n"
    "train_config"
)

if old_block not in text:
    raise SystemExit("Old block not found in cell 1")
text = text.replace(old_block, new_block)

# Add import torch after numpy
text = text.replace(
    "import numpy as np\n\n",
    "import numpy as np\n"
    "import torch\n\n",
)

# Split back to list of lines (with \n preserved for notebook)
nb["cells"][1]["source"] = [line + "\n" for line in text.split("\n")[:-1]] + (
    [text.split("\n")[-1]] if text.split("\n")[-1] else []
)
nb_path.write_text(json.dumps(nb, indent=2, ensure_ascii=False), encoding="utf-8")
print("Patched cell 1 OK")

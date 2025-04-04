from lm_eval import tasks
from lm_eval.tasks import get_task_dict, TaskManager
from datasets import load_dataset

# Could replace with just "leaderboard" but leaderboard_ifeval doesn't work for me on Zaratan
TASK_GROUPS = [
#    "leaderboard_ifeval",
    "leaderboard_bbh",
    "leaderboard_math_hard",
    "leaderboard_gpqa",
    "leaderboard_musr",
    "leaderboard_mmlu_pro"
]

task_manager = TaskManager()


for group in TASK_GROUPS:
    print(f"Downloading datasets for: {group}")
    _ = get_task_dict(group, task_manager)

MANUAL_DATASETS = {
    "DigitalLearningGmbH/MATH-lighteval": ['algebra', 'counting_and_probability', 'default', 'geometry', 'intermediate_algebra', 'number_theory', 'prealgebra', 'precalculus'],
    "TAUR-Lab/MuSR": ["default"],
    "TIGER-Lab/MMLU-Pro": ["default"]
}

print("\nManually downloading raw HF datasets for known missing sources...\n")

for dataset_name, configs in MANUAL_DATASETS.items():
    for config in configs:
        print(f"{dataset_name} - {config}")
        load_dataset(dataset_name, config)

"""
Evaluation script for comparing Vanilla GPT-2 and PPO-aligned GPT-2 outputs.

This script performs pairwise comparison between model responses and
counts preference wins using simple placeholder heuristics.

The evaluation logic is intentionally lightweight and designed to be
replaced with human or reward-model-based evaluation in future work.
"""

import pandas as pd
from pathlib import Path


def load_responses(csv_path: Path) -> pd.DataFrame:
    """
    Load prompt-response pairs from a CSV file.

    Expected CSV format:
        prompt,response
    """
    return pd.read_csv(csv_path)


def preference_heuristic(response_vanilla: str, response_ppo: str) -> int:
    """
    Placeholder preference function.

    Preference rules (simple heuristics):
    - Prefer longer non-empty responses
    - Return tie if lengths are similar

    Returns:
        1  -> vanilla preferred
        -1 -> PPO preferred
        0  -> tie
    """
    if not response_vanilla or not response_ppo:
        return 0

    len_v = len(response_vanilla.split())
    len_p = len(response_ppo.split())

    if abs(len_v - len_p) < 5:
        return 0

    return 1 if len_v > len_p else -1


def evaluate(vanilla_csv: Path, ppo_csv: Path) -> None:
    vanilla_df = load_responses(vanilla_csv)
    ppo_df = load_responses(ppo_csv)

    wins_vanilla = 0
    wins_ppo = 0
    ties = 0

    num_samples = min(len(vanilla_df), len(ppo_df))

    for i in range(num_samples):
        score = preference_heuristic(
            vanilla_df.loc[i, "response"],
            ppo_df.loc[i, "response"]
        )

        if score == 1:
            wins_vanilla += 1
        elif score == -1:
            wins_ppo += 1
        else:
            ties += 1

    print("\nEvaluation Results")
    print("------------------")
    print(f"Total comparisons: {num_samples}")
    print(f"Vanilla GPT-2 wins: {wins_vanilla}")
    print(f"PPO GPT-2 wins:     {wins_ppo}")
    print(f"Ties:               {ties}\n")


if __name__ == "__main__":
    vanilla_path = Path("outputs/vanilla_responses.csv")
    ppo_path = Path("outputs/ppo_responses.csv")

    if not vanilla_path.exists() or not ppo_path.exists():
        print(
            "Evaluation skipped: response files not found.\n"
            "Expected files:\n"
            " - outputs/vanilla_responses.csv\n"
            " - outputs/ppo_responses.csv\n"
        )
    else:
        evaluate(vanilla_path, ppo_path)


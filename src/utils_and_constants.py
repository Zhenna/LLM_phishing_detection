import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


SCORING = {
    "f1": f1_score,
    "precision": precision_score,
    "recall": recall_score,
    "accuracy": accuracy_score,
}


def save_scores(experiment: str, index: str, values: dict) -> None:
    """Log scores for individual models in the corresponding csv file"""
    llms = [
        "RoBERTa",
    ]
    models = ["NB", "LR", "KNN", "SVM", "XGBoost"]

    Path(f"outputs/scores/").mkdir(parents=True, exist_ok=True)

    file = Path(f"outputs/scores/{experiment}.csv")
    if file.is_file():
        scores = pd.read_csv(f"outputs/scores/{experiment}.csv", index_col=0)
        scores.loc[index] = values
    else:
        if index in llms:
            scores = pd.DataFrame(
                index=llms,
                columns=list(SCORING.keys()) + ["training_time", "inference_time"],
            )
        else:
            scores = pd.DataFrame(
                index=models,
                columns=list(SCORING.keys()) + ["training_time", "inference_time"],
            )
        scores.loc[index] = values

    scores.to_csv(f"outputs/scores/{experiment}.csv")
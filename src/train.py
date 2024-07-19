from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    TrainerCallback,
    AutoTokenizer,
    Trainer,
)

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate

import evaluate
import numpy as np
import copy
import time
import pandas as pd
import pickle

from src.get_data import get_raw_data, train_val_test_split
from src.utils_and_constants import (
    SCORING,
    save_scores,
)
from preprocess import transform_df, encode_df, tokenize, init_nltk


MODELS = {
    "NB": (MultinomialNB(), 300),
    "LR": (LogisticRegression(), 500),
    "KNN": (KNeighborsClassifier(n_neighbors=5), 150),
    "SVM": (SVC(kernel="sigmoid", gamma=1.0), 400),
    "XGBoost": (XGBClassifier(learning_rate=0.01, n_estimators=150), 350),
}


LLMS = {
    "RoBERTa": (
        AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=2
        ),
        AutoTokenizer.from_pretrained("roberta-base"),
    ),
    # other LLM and respective tokenizer
}


class EvalOnTrainCallback(TrainerCallback):
    """Custom callback to evaluate on the training set during training."""

    def __init__(self, trainer) -> None:
        super().__init__()
        self._trainer = trainer

    def on_epoch_end(self, args, state, control, **kwargs):
        if control.should_evaluate:
            control_train = copy.deepcopy(control)
            self._trainer.evaluate(
                eval_dataset=self._trainer.train_dataset, metric_key_prefix="train"
            )
            return control_train


def get_trainer(model, dataset):
    """Return a trainer object for transformer models."""

    def compute_metrics(y_pred):
        """Computer metrics during training."""
        logits, labels = y_pred
        predictions = np.argmax(logits, axis=-1)
        return evaluate.load("f1").compute(
            predictions=predictions, references=labels, average="macro"
        )

    training_args = TrainingArguments(
        output_dir="experiments",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=8,
        learning_rate=5e-5,
        num_train_epochs=10,  # use 1 epoch to debug
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        save_total_limit=2,  # 10, # takes space
        no_cuda=True
        # save_strategy = "no", # “epoch” or “steps”
        # evaluation_strategy = “epoch” or “steps”,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        compute_metrics=compute_metrics,
    )
    trainer.add_callback(EvalOnTrainCallback(trainer))
    return trainer


def predict(trainer, dataset):
    """Convert the predict function to specific classes to unify the API."""
    return trainer.predict(dataset).predictions.argmax(axis=-1)


def train_llms(
    label_col_name: str,
    text_col_name: str,
    seed=123,
    train_size=0.8,
    test_set="test",
    dataset_name="data.csv",
):
    """Train the large language model."""

    scores = pd.DataFrame(
        index=list(LLMS.keys()),
        columns=list(SCORING.keys()) + ["training_time", "inference_time"],
    )

    # Main loop
    df = get_raw_data(csv_name=dataset_name, label_col_name=label_col_name, text_col_name=text_col_name)
    _, dataset = train_val_test_split(df, train_size=train_size, has_val=False)

    # Name experiment
    experiment = f"llm_{dataset_name}_{test_set}_{train_size}_train_seed_{seed}"

    # Train, evaluate, test
    for model_name, (model, tokenizer) in LLMS.items():
        tokenized_dataset = tokenize(dataset, tokenizer)
        trainer = get_trainer(model, tokenized_dataset)

        # Train model
        start = time.time()
        trainer.train()
        end = time.time()

        # Save model
        trainer.save_model("outputs/model/roberta-trained")

        # Log Score
        scores.loc[model_name]["training_time"] = end - start
        log = pd.DataFrame(trainer.state.log_history)
        log.to_csv(f"outputs/csv/loss_{model_name}_{experiment}.csv")

        # Test model
        start = time.time()
        predictions = predict(trainer, tokenized_dataset[test_set])
        end = time.time()

        for score_name, score_fn in SCORING.items():
            scores.loc[model_name][score_name] = score_fn(
                dataset[test_set]["label"], predictions
            )

        scores.loc[model_name]["inference_time"] = end - start
        save_scores(experiment, model_name, scores.loc[model_name].to_dict())

    # Display scores
    print(scores)


def train_baselines(
    label_col_name: str,
    text_col_name: str,
    train_size=0.8,
    test_set="test",
    dataset_name="data.csv",
):
    """Train all the baseline models."""
    init_nltk()

    # Create list of metrics
    scores = pd.DataFrame(
        index=list(MODELS.keys()),
        columns=list(SCORING.keys()) + ["training_time", "inference_time"],
    )

    # Prepare data for training
    df = get_raw_data(
        csv_name=dataset_name, label_col_name=label_col_name, text_col_name=text_col_name
    )
    df = transform_df(df)
    (df_train, df_test), _ = train_val_test_split(
        df, train_size=train_size, has_val=False
    )

    # Name experiment
    experiment = f"ml_{dataset_name}_baseline"

    # Cross-validate and test every model
    for model_name, (model, max_iter) in MODELS.items():
        # Encode the dataset
        encoder = TfidfVectorizer(max_features=max_iter)
        X_train, y_train, encoder = encode_df(df_train, encoder)
        X_test, y_test, encoder = encode_df(df_test, encoder)

        # Evaluate model with cross-validation
        if test_set == "val":
            cv = cross_validate(
                model,
                X_train,
                y_train,
                scoring=list(SCORING.keys()),
                cv=5,
                n_jobs=-1,
            )
            for score_name, score_fn in SCORING.items():
                scores.loc[model_name][score_name] = cv[f"test_{score_name}"].mean()

        # Evaluate model on test set
        if test_set == "test":
            start = time.time()
            model.fit(X_train, y_train)
            end = time.time()
            scores.loc[model_name]["training_time"] = end - start

            start = time.time()
            y_pred = model.predict(X_test)
            end = time.time()

            scores.loc[model_name]["inference_time"] = end - start
            for score_name, score_fn in SCORING.items():
                scores.loc[model_name][score_name] = score_fn(y_pred, y_test)

        save_scores(experiment, model_name, scores.loc[model_name].to_dict())

        # save model
        pickle.dump(model, open(f"outputs/model/{model_name}.pkl", "wb"))

    # Display scores
    print(scores)

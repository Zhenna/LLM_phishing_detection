from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import pandas as pd
import pickle

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from src.preprocess import transform_df, encode_df, init_nltk
from src.get_data import get_raw_data, train_val_test_split

BEST_BASELINE_MODEL = {
    "NB": (MultinomialNB(), 300, "outputs/model/NB.pkl"),
    "LR": (LogisticRegression(), 500, "outputs/model/LR.pkl"),
    "KNN": (KNeighborsClassifier(n_neighbors=1), 150, "outputs/model/KNN.pkl"),
    "SVM": (SVC(kernel="sigmoid", gamma=1.0), 400, "outputs/model/SVM.pkl"),
    "XGBoost": (
        XGBClassifier(learning_rate=0.01, n_estimators=150),
        350,
        "outputs/model/XGBoost.pkl",
    ),
}

LLMS = {
    "RoBERTa": (
        AutoModelForSequenceClassification.from_pretrained(
            "outputs/model/roberta-trained"
        ),
        AutoTokenizer.from_pretrained("roberta-base"),
    ),
    # other LLM and respective tokenizer
}


class make_inference:

    def __init__(
        self,
        user_input: str,
        label_col_name: str,
        text_col_name: str,
        dataset_name="data.csv",
    ):
        self.text_input = user_input
        self.dataset_name = dataset_name
        self.label_col_name = label_col_name
        self.text_col_name = text_col_name

    def process_input(self):
        """convert text to matrix for baseline model only"""
        df_infer = pd.DataFrame(data={"text": self.text_input}, index=[0])
        return transform_df(df_infer)

    def best_baseline(self, train_size=0.8):
        """train (or load saved model) and predict"""

        init_nltk()

        # get data
        df = get_raw_data(
            csv_name=self.dataset_name,
            label_col_name=self.label_col_name,
            text_col_name=self.text_col_name,
        )
        df = transform_df(df)
        (df_train, _), _ = train_val_test_split(
            df, train_size=train_size, has_val=False
        )

        # get best model
        experiment = f"ml_{self.dataset_name}_baseline"
        df_score = pd.read_csv(f"outputs/scores/{experiment}.csv")
        model = df_score.iloc[df_score.f1.idxmax()][0]

        # vectorize data
        encoder = TfidfVectorizer(max_features=BEST_BASELINE_MODEL[model][1])
        _, _, encoder = encode_df(df_train, encoder)

        # load model
        model = pickle.load(open(BEST_BASELINE_MODEL[model][2], "rb"))

        # fit tokenizer to user input
        df = self.process_input()
        encoded_input = encoder.transform(df["transformed_text"]).toarray()

        # predict
        return model.predict(encoded_input)[0]

    def best_llm(self, model="RoBERTa"):
        """load trained model without training and make inference directly"""

        # load model and tokenizer
        trainer = LLMS[model][0]
        tokenizer = LLMS[model][1]

        # tokenize input
        tokenized_input = tokenizer(
            self.text_input,
            truncation=True,
            is_split_into_words=False,
            return_tensors="pt",
        )

        # predict
        output = trainer(tokenized_input["input_ids"])
        prediction = output.logits.argmax(axis=-1)

        return prediction.item()

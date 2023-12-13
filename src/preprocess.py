import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def init_nltk():
    nltk.download("punkt")
    nltk.download('stopwords')


def tokenize_words(text):
    """Tokenize words in text and remove punctuation"""
    text = word_tokenize(str(text).lower())
    text = [token for token in text if token.isalnum()]
    return text


def remove_stopwords(text):
    """Remove stopwords from the text"""
    text = [token for token in text if token not in stopwords.words("english")]
    return text


def stem(text):
    """Stem the text (originate => origin)"""
    text = [ps.stem(token) for token in text]
    return text


def transform(text):
    """Tokenize, remove stopwords, stem the text"""
    text = tokenize_words(text)
    text = remove_stopwords(text)
    text = stem(text)
    text = " ".join(text)
    return text


def transform_df(df, text_col_name: str='text'):
    """Apply the transform function to the dataframe"""
    df["transformed_text"] = df[text_col_name].apply(transform)
    return df


def encode_df(df, encoder, label_col_name: str="label"):
    """Encode the features for training set"""
    if hasattr(encoder, "vocabulary_"):
        X = encoder.transform(df["transformed_text"]).toarray()
    else:
        X = encoder.fit_transform(df["transformed_text"]).toarray()
    y = df[label_col_name].values
    return X, y, encoder


def tokenize(dataset, tokenizer, text_col="text"):
    """Tokenize dataset"""

    def tokenization(examples):
        return tokenizer(examples[text_col], padding="max_length", truncation=True)

    if tokenizer is None:
        return dataset

    else:
        return dataset.map(tokenization, batched=True)
"""Column detection and word splitting shared by EDA keywords and row filters.

EDA counts how many texts in a class contain a term; the row browser filters
to those texts. Both read from here so the count and the filtered total agree.
"""

from sklearn.feature_extraction.text import CountVectorizer

LABEL_COL_NAMES = ["label", "labels", "class", "category", "target"]
TEXT_COL_NAMES = ["text", "content", "document", "sentence"]


def find_column(df, names):
    """The first column whose lowercased name is in `names`, or None."""
    for col in df.columns:
        if str(col).lower() in names:
            return col
    return None


def keyword_vectorizer():
    """Binary unigram counts without English stop words, capped at 2000 terms."""
    return CountVectorizer(
        stop_words="english", ngram_range=(1, 1), max_features=2000, binary=True
    )


def text_values(df, text_col):
    # astype(str) before fillna, so a missing text reads as "nan" in both places.
    return df[text_col].astype(str).fillna("")


def label_values(df, label_col):
    return df[label_col].astype(str)


def texts_containing(texts, term):
    """Boolean mask of the texts whose words include `term` (case-insensitive)."""
    analyzer = CountVectorizer().build_analyzer()
    term = str(term).lower()
    return texts.map(lambda text: term in set(analyzer(text)))

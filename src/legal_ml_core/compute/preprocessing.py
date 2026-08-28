import re

import nltk


def _ensure_nltk_data():
    for pkg in ("punkt", "stopwords"):
        try:
            nltk.data.find(
                f"tokenizers/{pkg}" if pkg == "punkt" else f"corpora/{pkg}"
            )
        except LookupError:
            nltk.download(pkg, quiet=True)


class TextPreprocessor:
    def __init__(self, options):
        self.options = options
        _ensure_nltk_data()
        self.stop_words = set(nltk.corpus.stopwords.words("english"))

    def clean_text(self, text):
        if not isinstance(text, str):
            return ""

        # Remove HTML tags
        if self.options.get("remove_html", True):
            text = re.sub(r"<[^>]+>", "", text)

        # Remove URLs
        if self.options.get("remove_urls", True):
            text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

        # Remove emails
        if self.options.get("remove_emails", True):
            text = re.sub(r"\S+@\S+", "", text)

        # Remove numbers
        if self.options.get("remove_numbers", False):
            text = re.sub(r"\d+", "", text)

        # Remove special characters
        if self.options.get("remove_special_chars", False):
            text = re.sub(r"[^a-zA-Z\s]", "", text)

        # Convert to lowercase
        if self.options.get("lowercase", True):
            text = text.lower()

        return text.strip()

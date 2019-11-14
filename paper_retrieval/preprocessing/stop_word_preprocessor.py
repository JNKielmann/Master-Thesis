from spacy.lang.en.stop_words import STOP_WORDS


class StopWordPreprocessor():
    def __init__(self):
        self.name = "NoStopWords"

    def __call__(self, text: str):
        return " ".join([word for word in text.split(" ") if
                         word not in STOP_WORDS and len(word) > 2])

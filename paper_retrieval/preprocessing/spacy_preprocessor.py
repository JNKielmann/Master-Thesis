import spacy


class SpacyPreprocessor:
    def __init__(self, lemmatization: str = "all"):
        if lemmatization != "all" and lemmatization != "nouns":
            raise ValueError("lemmatization must be 'all' or 'nouns'")
        self.lemmatization = lemmatization
        self.name = "spacy_lemmatization_" + lemmatization
        self.spacy_parser = None

    def __call__(self, text: str):
        if not self.spacy_parser:
            self.spacy_parser = spacy.load('en', disable=['ner'])
        tokens = self.spacy_parser(text)
        if self.lemmatization == "all":
            return " ".join([token.lemma_ for token in tokens])
        else:
            return " ".join([token.lemma_ if token.pos_ == "NOUN" else token.text
                             for token in tokens])

    def __getstate__(self):
        return self.name, self.lemmatization

    def __setstate__(self, state):
        self.name, self.lemmatization = state
        self.spacy_parser = None

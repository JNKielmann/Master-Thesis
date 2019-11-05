import spacy

class SpacyPreprocessor:
    def __init__(self, lemmatization: bool=False, combine_noun_chunks:bool=False):
        if not lemmatization and not combine_noun_chunks:
            raise ValueError("lemmatization and combine_noun_chunks of SpacyPreprocessor are False")
        self.lemmatization = lemmatization
        self.combine_noun_chunks = combine_noun_chunks
        self.name = "spacy"
        if self.lemmatization:
            self.name += "_lemmatization"
        if self.combine_noun_chunks:
            self.name += "_nounchunks"
        self.spacy_parser = None

    def __call__(self, text: str):
        if not self.spacy_parser:
            self.spacy_parser = spacy.load('en', disable=['ner'])
        tokens = self.spacy_parser(text)
        if self.combine_noun_chunks:
            for noun_phrase in list(tokens.noun_chunks):
                noun_phrase.merge(lemma=noun_phrase.lemma_)
        return " ".join([token.lemma_ if token.pos_ == "NOUN" else token.text for token in tokens])

    def __getstate__(self):
        return self.name, self.lemmatization, self.combine_noun_chunks

    def __setstate__(self, state):
        self.name, self.lemmatization, self.combine_noun_chunks = state
        self.spacy_parser = None

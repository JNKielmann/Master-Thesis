import re


class BasicPreprocessing:
    def __init__(self):
        self.name = "basic"

    def __call__(self, text: str):
        return clean_text(text)


def clean_text(text):
    text = text.lower()
    text = replace_all(text, [
        ("n't ", " not "),
        ("'ve ", " have "),
        ("'ll ", " will "),
        ("'s ", " "),
        ("'d ", " ")
    ])
    text = sub_iso(text)
    text = sub_unknown_chars(text)
    text = sub_numbers(text)
    text = sub_multiple_spaces(text)
    return text


def sub_iso(text):
    return re.sub(
        r"\biso (\d+)(-(\d+))?\b",
        lambda m: f"iso_{m.group(1)}{'_' + m.group(3) if m.group(3) else ''}",
        text)


def sub_unknown_chars(text):
    return re.sub(r"([^a-z0-9_ ])+", " ", text)


def sub_numbers(text):
    return re.sub(r"\b(\d+)\b", " ", text)


def sub_multiple_spaces(text):
    return re.sub(r"\s\s+", " ", text)


def replace_all(text, replacement_list):
    for old, new in replacement_list:
        text = text.replace(old, new)
    return text

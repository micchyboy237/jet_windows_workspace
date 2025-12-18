import re

def split_words(text: str) -> list[str]:
    # Updated regex to handle cases like "A.F.&A.M."
    # This pattern matches words that start and end with an alphanumeric character,
    # including words with hyphens, apostrophes, periods, and ampersands in the middle
    return re.findall(r"(\b[\w'.&-]+\b)", text)


def split_tokenize_text(text: str) -> list[str]:
    # Split on whitespace and punctuation, except for apostrophes within words
    tokens = re.findall(r"\b\w+(?:'\w+)?\b|\b\d+\b|[^\w\s]", text)
    # Find contraction tokens and split them into separate tokens
    contractions = re.compile(r"(?<=\w)'(?=\w)")
    tokens = [token for token in tokens if token]
    for token_idx, token in enumerate(tokens):
        if contractions.search(token):
            # Split the token into separate tokens but the right token has the apostrophe prefix
            splitted_tokens = contractions.split(token)
            for i, token in enumerate(splitted_tokens):
                if i > 0:
                    splitted_tokens[i] = "'" + token

            # Remove the original token and insert the splitted tokens
            tokens.pop(token_idx)
            tokens[token_idx:token_idx] = splitted_tokens

    return tokens


def remove_non_alpha_numeric(text: str) -> str:
    splitted_words = split_words(text)
    return " ".join(splitted_words)
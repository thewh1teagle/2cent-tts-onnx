import tokenizers

class Tokenizer:
    def __init__(self, path: str):
        self.tokenizer = tokenizers.Tokenizer.from_file(path)

    def tokenize(self, text: str):
        # Encode the text and return tokens as strings
        encoded = self.tokenizer.encode(text)
        return encoded.tokens

    def tokenize_ids(self, text: str):
        # Encode the text and return token IDs
        encoded = self.tokenizer.encode(text)
        return encoded.ids

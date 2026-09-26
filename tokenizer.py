import os
from tokenizers import Tokenizer

def load_tokenizer(path: str = "tokenizer.json") -> Tokenizer:
    target = path
    if os.path.isdir(target):
        target = os.path.join(target, "tokenizer.json")
    if not os.path.exists(target):
        raise FileNotFoundError(f"Tokenizer file not found: {path}")
    return Tokenizer.from_file(target)

load_stock_tokenizer = load_tokenizer

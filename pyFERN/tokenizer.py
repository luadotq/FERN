import os
from typing import List, Optional, Union, Any
import numpy as np

class TokenList(list):
    @property
    def ids(self) -> List[int]:
        return self


class FERNTokenizer:
    def __init__(self, backend_tokenizer: Any, name_or_path: str = ""):
        self._tokenizer = backend_tokenizer
        self.name_or_path = name_or_path
        self._is_hf = hasattr(backend_tokenizer, "batch_decode")

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "FERNTokenizer":
        path = str(pretrained_model_name_or_path)

        try:
            from transformers import AutoTokenizer
            try:
                kw = dict(kwargs)
                kw.setdefault("model_max_length", int(1e9))
                hf_tok = AutoTokenizer.from_pretrained(path, **kw)
                return cls(hf_tok, name_or_path=path)
            except Exception:
                pass
        except ImportError:
            pass

        try:
            from tokenizers import Tokenizer
            if os.path.isfile(path):
                raw_tok = Tokenizer.from_file(path)
                return cls(raw_tok, name_or_path=path)
            if os.path.isdir(path):
                tok_file = os.path.join(path, "tokenizer.json")
                if os.path.exists(tok_file):
                    raw_tok = Tokenizer.from_file(tok_file)
                    return cls(raw_tok, name_or_path=path)
            try:
                raw_tok = Tokenizer.from_pretrained(path)
                return cls(raw_tok, name_or_path=path)
            except Exception:
                pass
        except ImportError:
            pass

        raise FileNotFoundError(
            f"Could not load tokenizer from '{path}'. Make sure it is a valid HF Hub model ID, "
            "a local directory with tokenizer config, or a valid tokenizer.json file."
        )

    def encode(self, text: str, add_special_tokens: bool = True) -> TokenList:
        if self._is_hf:
            ids = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return TokenList(ids)
        else:
            enc = self._tokenizer.encode(text, add_special_tokens=add_special_tokens)
            return TokenList(enc.ids)

    def batch_encode(self, texts: List[str], add_special_tokens: bool = True) -> List[TokenList]:
        raw = getattr(self._tokenizer, "_tokenizer", self._tokenizer)
        if hasattr(raw, "encode_batch"):
            encs = raw.encode_batch(texts, add_special_tokens=add_special_tokens)
            return [TokenList(enc.ids) for enc in encs]
        return [self.encode(t, add_special_tokens=add_special_tokens) for t in texts]

    encode_batch = batch_encode

    def decode(
        self,
        token_ids: Union[List[int], np.ndarray, Any],
        skip_special_tokens: bool = True,
    ) -> str:
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, np.ndarray):
            token_ids = token_ids.tolist()

        if self._is_hf:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        else:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    @property
    def vocab_size(self) -> int:
        if hasattr(self._tokenizer, "vocab_size"):
            return self._tokenizer.vocab_size
        if hasattr(self._tokenizer, "get_vocab_size"):
            return self._tokenizer.get_vocab_size()
        return len(self._tokenizer)

    def get_vocab_size(self) -> int:
        return self.vocab_size

    @property
    def pad_token_id(self) -> Optional[int]:
        if hasattr(self._tokenizer, "pad_token_id"):
            return self._tokenizer.pad_token_id
        return getattr(self._tokenizer, "token_to_id", lambda x: None)("<pad>")

    @property
    def bos_token_id(self) -> Optional[int]:
        if hasattr(self._tokenizer, "bos_token_id"):
            return self._tokenizer.bos_token_id
        return getattr(self._tokenizer, "token_to_id", lambda x: None)("<s>")

    @property
    def eos_token_id(self) -> Optional[int]:
        if hasattr(self._tokenizer, "eos_token_id"):
            return self._tokenizer.eos_token_id
        return getattr(self._tokenizer, "token_to_id", lambda x: None)("</s>")

    @property
    def unk_token_id(self) -> Optional[int]:
        if hasattr(self._tokenizer, "unk_token_id"):
            return self._tokenizer.unk_token_id
        return getattr(self._tokenizer, "token_to_id", lambda x: None)("<unk>")

    def token_to_id(self, token: str) -> Optional[int]:
        if hasattr(self._tokenizer, "token_to_id"):
            return self._tokenizer.token_to_id(token)
        if hasattr(self._tokenizer, "convert_tokens_to_ids"):
            return self._tokenizer.convert_tokens_to_ids(token)
        return None

    def id_to_token(self, token_id: int) -> Optional[str]:
        if hasattr(self._tokenizer, "id_to_token"):
            return self._tokenizer.id_to_token(token_id)
        if hasattr(self._tokenizer, "convert_ids_to_tokens"):
            return self._tokenizer.convert_ids_to_tokens(token_id)
        return None

    def save_pretrained(self, save_directory: str):
        os.makedirs(save_directory, exist_ok=True)
        if hasattr(self._tokenizer, "save_pretrained"):
            self._tokenizer.save_pretrained(save_directory)
        elif hasattr(self._tokenizer, "save"):
            self._tokenizer.save(os.path.join(save_directory, "tokenizer.json"))

    @classmethod
    def train_bpe(
        cls,
        files: List[str],
        vocab_size: int = 32000,
        save_path: Optional[str] = None,
        special_tokens: Optional[List[str]] = None,
    ) -> "FERNTokenizer":
        from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, trainers, decoders

        special = special_tokens or ["<pad>", "<unk>", "<s>", "</s>"]
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        )
        tok.train(files, trainer=trainer)

        if save_path:
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            tok.save(save_path)

        return cls(tok, name_or_path=save_path or "custom_bpe")


def load_tokenizer(path: str = "tokenizer.json") -> FERNTokenizer:
    return FERNTokenizer.from_pretrained(path)

load_stock_tokenizer = load_tokenizer

import os
import glob
import json
import math
import re
from typing import List, Optional, Union, Tuple, Iterator, Any
import numpy as np
import torch

class PretokenizedDataset:
    def __init__(
        self,
        path_or_pattern: str,
        seq_len: int = 2048,
        dtype: Union[np.dtype, str] = np.uint16,
        rank: int = 0,
        world_size: int = 1,
    ):
        # Auto-detect dtype from metadata.json if in directory
        if isinstance(path_or_pattern, str) and os.path.isdir(path_or_pattern):
            meta_file = os.path.join(path_or_pattern, "metadata.json")
            if os.path.exists(meta_file):
                try:
                    with open(meta_file, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        if "dtype" in meta and (dtype == np.uint16 or dtype == "uint16"):
                            dtype = meta["dtype"]
                except Exception:
                    pass

        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.rank = rank
        self.world_size = world_size

        # Resolve files
        if isinstance(path_or_pattern, str) and os.path.isdir(path_or_pattern):
            patterns = [
                os.path.join(path_or_pattern, "*.bin"),
                os.path.join(path_or_pattern, "*.npy"),
            ]
            files = []
            for p in patterns:
                files.extend(sorted(glob.glob(p)))
            if not files:
                # Check for train.bin specifically
                default_bin = os.path.join(path_or_pattern, "train.bin")
                if os.path.exists(default_bin):
                    files = [default_bin]
        elif isinstance(path_or_pattern, (list, tuple)):
            files = [f for f in path_or_pattern if os.path.exists(f)]
        else:
            files = sorted(glob.glob(path_or_pattern))
            if not files and os.path.exists(path_or_pattern):
                files = [path_or_pattern]

        if not files:
            raise FileNotFoundError(f"No binary dataset files found matching: {path_or_pattern}")

        self.files = files
        self.mmaps: List[np.memmap] = []
        self.shard_lengths: List[int] = []

        itemsize = self.dtype.itemsize
        for f in self.files:
            file_size = os.path.getsize(f)
            num_tokens = file_size // itemsize
            if num_tokens <= self.seq_len + 1:
                continue
            m = np.memmap(f, dtype=self.dtype, mode="r")
            self.mmaps.append(m)
            self.shard_lengths.append(num_tokens)

        if not self.mmaps:
            raise ValueError(
                f"Found files {self.files}, but none have enough tokens for seq_len={seq_len}."
            )

        self.total_tokens = sum(self.shard_lengths)
        # Number of non-overlapping sequences per shard
        self.shard_samples = [(n - 1) // self.seq_len for n in self.shard_lengths]
        self.total_samples = sum(self.shard_samples)

        # Build cumulative offsets for O(1) sample index lookup
        self.sample_cum = np.cumsum([0] + self.shard_samples)

    def __len__(self) -> int:
        if self.world_size <= 1:
            return self.total_samples
        # Distributed sharding: number of samples assigned to this rank
        return (self.total_samples + self.world_size - 1 - self.rank) // self.world_size

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset with {len(self)} samples")

        # Map rank local index to global sample index
        if self.world_size > 1:
            global_idx = idx * self.world_size + self.rank
        else:
            global_idx = idx

        # Find which shard contains global_idx using binary search
        shard_idx = int(np.searchsorted(self.sample_cum, global_idx, side="right") - 1)
        local_sample_idx = global_idx - self.sample_cum[shard_idx]

        token_offset = local_sample_idx * self.seq_len
        m = self.mmaps[shard_idx]

        chunk = m[token_offset : token_offset + self.seq_len + 1].astype(np.int32)
        inputs = chunk[: self.seq_len]
        targets = chunk[1 : self.seq_len + 1]
        return inputs, targets

    def get_batch(self, batch_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bs = len(batch_indices)
        inputs = np.empty((bs, self.seq_len), dtype=np.int32)
        targets = np.empty((bs, self.seq_len), dtype=np.int32)

        for i, idx in enumerate(batch_indices):
            inp, tgt = self[idx]
            inputs[i] = inp
            targets[i] = tgt

        return inputs, targets


class JAXDataIterator:
    def __init__(
        self,
        dataset: PretokenizedDataset,
        batch_size: int = 16,
        shuffle: bool = True,
        seed: int = 42,
        infinite: bool = True,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.infinite = infinite
        self.rng = np.random.RandomState(seed)
        self.indices = np.arange(len(self.dataset))
        self.pos = 0

        if self.shuffle:
            self.rng.shuffle(self.indices)

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.pos + self.batch_size > len(self.indices):
            if not self.infinite:
                raise StopIteration
            if self.shuffle:
                self.rng.shuffle(self.indices)
            self.pos = 0

        batch_idx = self.indices[self.pos : self.pos + self.batch_size]
        self.pos += self.batch_size

        return self.dataset.get_batch(batch_idx)


def prepare_pretokenized_dataset(
    input_files: Union[str, List[str]],
    output_dir: str,
    tokenizer,
    seq_len: int = 2048,
    dtype: str = "uint16",
    max_tokens_per_shard: int = 100_000_000,
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    np_dtype = np.uint16 if dtype == "uint16" else np.uint32

    if isinstance(input_files, str):
        if os.path.isdir(input_files):
            input_files = sorted(glob.glob(os.path.join(input_files, "*.*")))
        else:
            input_files = sorted(glob.glob(input_files))

    shard_idx = 0
    total_tokens = 0
    current_tokens: List[int] = []

    def flush_shard():
        nonlocal shard_idx, current_tokens
        if not current_tokens:
            return
        shard_path = os.path.join(output_dir, f"shard_{shard_idx:05d}.bin")
        arr = np.array(current_tokens, dtype=np_dtype)
        arr.tofile(shard_path)
        current_tokens = []
        shard_idx += 1

    for file_path in input_files:
        if not os.path.exists(file_path):
            continue
        ext = os.path.splitext(file_path)[1].lower()
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            if ext == ".jsonl":
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        text = record.get("text", record.get("content", ""))
                    except Exception:
                        text = line
                    if text:
                        toks = tokenizer.encode(text)
                        if hasattr(toks, "ids"):
                            toks = toks.ids
                        current_tokens.extend(toks)
                        total_tokens += len(toks)
                        if len(current_tokens) >= max_tokens_per_shard:
                            flush_shard()
            elif ext == ".json":
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    text = item.get("text", "") if isinstance(item, dict) else str(item)
                    if text:
                        toks = tokenizer.encode(text)
                        if hasattr(toks, "ids"):
                            toks = toks.ids
                        current_tokens.extend(toks)
                        total_tokens += len(toks)
                        if len(current_tokens) >= max_tokens_per_shard:
                            flush_shard()
            else:
                text = f.read()
                if text:
                    toks = tokenizer.encode(text)
                    if hasattr(toks, "ids"):
                        toks = toks.ids
                    current_tokens.extend(toks)
                    total_tokens += len(toks)
                    if len(current_tokens) >= max_tokens_per_shard:
                        flush_shard()

    flush_shard()

    metadata = {
        "total_tokens": total_tokens,
        "shards": shard_idx,
        "dtype": dtype,
        "seq_len": seq_len,
        "output_dir": output_dir,
    }
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return metadata


class LossPattern:
    def __init__(
        self,
        mode: str = "pattern",
        patterns: Optional[Union[str, Tuple[str, Optional[str]], List[Any]]] = None,
        regex: Optional[Union[str, List[str]]] = None,
        ignore_idx: int = -100,
    ):
        self.mode = mode
        self.ignore_idx = ignore_idx
        self.regexes = []
        if regex:
            r_list = [regex] if isinstance(regex, str) else regex
            self.regexes.extend([re.compile(r, re.DOTALL) for r in r_list])
        elif patterns:
            p_list = [patterns] if isinstance(patterns, (str, tuple)) else patterns
            for p in p_list:
                if isinstance(p, tuple) and len(p) == 2 and p[1]:
                    self.regexes.append(
                        re.compile(re.escape(p[0]) + r"(.*?)(?=" + re.escape(p[1]) + r"|$)", re.DOTALL)
                    )
                else:
                    start = p[0] if isinstance(p, (tuple, list)) else p
                    self.regexes.append(
                        re.compile(re.escape(start) + r"(.*)", re.DOTALL)
                    )
        else:
            self.regexes.append(
                re.compile(r"(?:Assistant:|Ответ:)\s*(.*?)(?=\n(?:User:|Вопрос:)|$)", re.DOTALL)
            )

    def mask(self, text: str, enc, input_ids: torch.Tensor) -> torch.Tensor:
        if self.mode == "full":
            return input_ids.clone()
        labels = torch.full_like(input_ids, self.ignore_idx)
        spans = []
        for rx in self.regexes:
            for m in rx.finditer(text):
                g = 1 if rx.groups >= 1 else 0
                spans.append((m.start(g), m.end(g)))
        if not spans:
            return input_ids.clone()

        if hasattr(enc, "offsets"):
            for s_start, s_end in spans:
                for idx, (c_start, c_end) in enumerate(enc.offsets):
                    if idx < len(labels) and c_start >= s_start and c_end <= s_end and c_start < c_end:
                        labels[idx] = input_ids[idx]
        else:
            return input_ids.clone()

        if (labels != self.ignore_idx).sum() == 0:
            return input_ids.clone()
        return labels


class SFTDataset:
    def __init__(
        self,
        path: str,
        tok=None,
        max_len: Optional[int] = None,
        loss_pattern: Optional[Union[LossPattern, dict, str, list]] = None,
        loss_mode: str = "pattern",
        sep_pattern: Union[str, List[str], None] = None,
        ignore_index: int = -100,
    ):
        self.tok = tok
        self.max_len = max_len if (max_len is not None and max_len > 0) else None
        self.ignore_index = ignore_index

        if isinstance(loss_pattern, LossPattern):
            self.loss_pattern = loss_pattern
        elif isinstance(loss_pattern, dict):
            self.loss_pattern = LossPattern(**loss_pattern)
        elif isinstance(loss_pattern, (str, list, tuple)):
            self.loss_pattern = LossPattern(patterns=loss_pattern, ignore_idx=ignore_index)
        else:
            patterns = sep_pattern if sep_pattern is not None else None
            self.loss_pattern = LossPattern(mode=loss_mode, patterns=patterns, ignore_idx=ignore_index)

        self.data = []
        self.lengths = []

        if path.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(path)
            self.data = df.to_dict(orient="records")
        else:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        self.data.append(json.loads(line))

        for r in self.data:
            if "length" in r:
                self.lengths.append(int(r["length"]))
            elif "input_ids" in r:
                self.lengths.append(len(r["input_ids"]))
            else:
                txt = r.get("text", "")
                self.lengths.append(max(1, len(txt) // 4))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data[idx]
        txt = row.get("text", "")

        if "input_ids" in row and "labels" in row:
            tids = row["input_ids"] if self.max_len is None else row["input_ids"][: self.max_len]
            lbls = row["labels"] if self.max_len is None else row["labels"][: self.max_len]
            return torch.tensor(tids, dtype=torch.long), torch.tensor(lbls, dtype=torch.long)

        if self.tok is None:
            raise ValueError("Tokenizer required when dataset contains un-tokenized text.")
        enc = self.tok.encode(txt, add_special_tokens=False) if hasattr(self.tok, "encode") else None
        tokens = enc.ids if hasattr(enc, "ids") else (enc if isinstance(enc, list) else [])
        if self.max_len is not None and len(tokens) > self.max_len:
            tokens = tokens[: self.max_len]

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = self.loss_pattern.mask(txt, enc, input_ids)
        return input_ids, labels


class BucketBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        lengths: List[int],
        batch_size: int = 4,
        boundaries: Optional[List[int]] = None,
        shuffle: bool = True,
        max_tokens_per_batch: Optional[int] = None,
        seed: int = 42,
    ):
        self.boundaries = boundaries or [2048, 4096, 8192, 16384, 32768, 65536]
        self.batch_size = batch_size
        self.max_tokens = max_tokens_per_batch
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self.lengths = lengths
        self.batches = self._build_batches()

    def set_epoch(self, epoch: int):
        self.epoch = epoch
        self.batches = self._build_batches()

    def _build_batches(self):
        import bisect
        buckets = [[] for _ in range(len(self.boundaries) + 1)]
        for idx, l in enumerate(self.lengths):
            b_idx = bisect.bisect_right(self.boundaries, l)
            buckets[b_idx].append(idx)

        rng = torch.Generator().manual_seed(self.seed + self.epoch)
        batches = []
        for b_idx, idxs in enumerate(buckets):
            if not idxs:
                continue
            if self.shuffle:
                p = torch.randperm(len(idxs), generator=rng).tolist()
                idxs = [idxs[i] for i in p]

            b_len = self.boundaries[min(b_idx, len(self.boundaries) - 1)]
            bs = max(1, self.max_tokens // b_len) if self.max_tokens else self.batch_size
            for i in range(0, len(idxs), bs):
                batches.append(idxs[i : i + bs])

        if self.shuffle:
            p = torch.randperm(len(batches), generator=rng).tolist()
            batches = [batches[i] for i in p]
        return batches

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


def sft_collate_fn(batch):
    max_l = max(len(x[0]) for x in batch)
    b_size = len(batch)
    inputs = torch.zeros(b_size, max_l, dtype=torch.long)
    labels = torch.full((b_size, max_l), -100, dtype=torch.long)
    for i, (inp, lbl) in enumerate(batch):
        inputs[i, : len(inp)] = inp
        labels[i, : len(lbl)] = lbl
    return inputs, labels


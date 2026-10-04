import os
import glob
import json
import math
from typing import List, Optional, Union, Tuple, Iterator
import numpy as np

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
        if os.path.isdir(path_or_pattern):
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
        if os.path.isdir(path_or_pattern):
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

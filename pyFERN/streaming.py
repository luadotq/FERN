import os
import glob
import json
import shutil
import queue
import threading
from typing import Optional, List, Union, Tuple, Iterator
import numpy as np

from .data import PretokenizedDataset, JAXDataIterator

class SlidingShardLoader:
    def __init__(
        self,
        dataset_path_or_repo: str,
        cache_dir: str = "/tmp/fern_shards",
        batch_size: int = 16,
        seq_len: int = 2048,
        dtype: Union[str, np.dtype] = np.uint32,
        window_size: int = 2,
        val_ratio: float = 0.05,
        hf_token: Optional[str] = None,
        infinite: bool = True,
    ):
        self.source = dataset_path_or_repo
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.dtype = np.dtype(dtype)
        self.window_size = max(1, window_size)
        self.val_ratio = val_ratio
        self.hf_token = hf_token
        self.infinite = infinite

        self.is_hf = not os.path.exists(self.source) and ("/" in self.source)
        self.train_dir = os.path.join(cache_dir, "train")
        self.val_dir = os.path.join(cache_dir, "val")
        os.makedirs(self.train_dir, exist_ok=True)
        os.makedirs(self.val_dir, exist_ok=True)

        self.train_shards: List[str] = []
        self.val_shards: List[str] = []
        self.val_dataset: Optional[PretokenizedDataset] = None

        self._discover_and_split()

        self.download_queue: queue.Queue = queue.Queue(maxsize=self.window_size)
        self.stopped = threading.Event()
        self.fetch_thread = None

        if self.is_hf:
            self.fetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
            self.fetch_thread.start()

        self.current_iter: Optional[JAXDataIterator] = None
        self.current_shard_path: Optional[str] = None

    def _discover_and_split(self):
        if self.is_hf:
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi(token=self.hf_token)
            files = api.list_repo_files(repo_id=self.source, repo_type="dataset")

            # Try downloading metadata.json
            if "metadata.json" in files:
                try:
                    meta_path = hf_hub_download(
                        repo_id=self.source,
                        filename="metadata.json",
                        repo_type="dataset",
                        token=self.hf_token,
                        local_dir=self.cache_dir,
                    )
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        if "dtype" in meta:
                            self.dtype = np.dtype(meta["dtype"])
                        if "seq_len" in meta:
                            self.seq_len = int(meta["seq_len"])
                except Exception:
                    pass

            bin_files = sorted([f for f in files if f.endswith(".bin")])
            if not bin_files:
                raise FileNotFoundError(f"No .bin shards found in HF repo '{self.source}'")

            val_count = max(1, int(len(bin_files) * self.val_ratio)) if self.val_ratio > 0 else 0
            if val_count > 0:
                self.val_shards = bin_files[-val_count:]
                self.train_shards = bin_files[:-val_count]
            else:
                self.train_shards = bin_files
                self.val_shards = []

            # Download validation shards once
            if self.val_shards:
                for vf in self.val_shards:
                    hf_hub_download(
                        repo_id=self.source,
                        filename=vf,
                        repo_type="dataset",
                        token=self.hf_token,
                        local_dir=self.val_dir,
                    )
                self.val_dataset = PretokenizedDataset(self.val_dir, seq_len=self.seq_len, dtype=self.dtype)

        else:
            if os.path.isdir(self.source):
                bin_files = sorted(glob.glob(os.path.join(self.source, "*.bin")))
            else:
                bin_files = sorted(glob.glob(self.source))

            if not bin_files:
                raise FileNotFoundError(f"No .bin shards found in '{self.source}'")

            val_count = max(1, int(len(bin_files) * self.val_ratio)) if self.val_ratio > 0 else 0
            if val_count > 0:
                self.val_shards = bin_files[-val_count:]
                self.train_shards = bin_files[:-val_count]
                self.val_dataset = PretokenizedDataset(self.val_shards, seq_len=self.seq_len, dtype=self.dtype)
            else:
                self.train_shards = bin_files

    def _prefetch_loop(self):
        from huggingface_hub import hf_hub_download
        while not self.stopped.is_set():
            for shard_name in self.train_shards:
                if self.stopped.is_set():
                    break
                try:
                    local_file = hf_hub_download(
                        repo_id=self.source,
                        filename=shard_name,
                        repo_type="dataset",
                        token=self.hf_token,
                        local_dir=self.train_dir,
                    )
                    self.download_queue.put(local_file)
                except Exception as e:
                    print(f"[Prefetch Warning] Failed to download {shard_name}: {e}")
                    time.sleep(2.0)
            if not self.infinite:
                break
        self.download_queue.put(None)

    def _load_next_shard(self) -> bool:
        # Delete previous shard from disk if streamed from HF
        if self.is_hf and self.current_shard_path and os.path.exists(self.current_shard_path):
            try:
                os.remove(self.current_shard_path)
            except OSError:
                pass

        if self.is_hf:
            shard_path = self.download_queue.get()
            if shard_path is None:
                return False
            self.current_shard_path = shard_path
            ds = PretokenizedDataset(shard_path, seq_len=self.seq_len, dtype=self.dtype)
            self.current_iter = JAXDataIterator(ds, batch_size=self.batch_size, shuffle=True, infinite=False)
            return True
        else:
            if not hasattr(self, "_local_shard_idx"):
                self._local_shard_idx = 0
            if self._local_shard_idx >= len(self.train_shards):
                if not self.infinite:
                    return False
                self._local_shard_idx = 0
            shard_path = self.train_shards[self._local_shard_idx]
            self._local_shard_idx += 1
            self.current_shard_path = shard_path
            ds = PretokenizedDataset(shard_path, seq_len=self.seq_len, dtype=self.dtype)
            self.current_iter = JAXDataIterator(ds, batch_size=self.batch_size, shuffle=True, infinite=False)
            return True

    def __iter__(self) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        return self

    def __next__(self) -> Tuple[np.ndarray, np.ndarray]:
        while True:
            if self.current_iter is None:
                if not self._load_next_shard():
                    raise StopIteration
            try:
                return next(self.current_iter)
            except StopIteration:
                self.current_iter = None

    def cleanup(self):
        self.stopped.set()
        if self.is_hf and os.path.exists(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)

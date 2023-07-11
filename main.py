"""
Minimalistic replication of funny behavior.
"""

from argparse import ArgumentParser
import os
from pathlib import Path
import psutil
from typing import Optional

from datasets import concatenate_datasets, Dataset
import pandas as pd
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast


DATA_PATH = Path("./data")
LABELS_PATH = Path("./trainLabels.csv")
TOKENIZER_PATH = Path("./tokenizer.json")

SPECIALS = {
    "unk_token": "<unk>",
    "pad_token": "<pad>",
    "mask_token": "<mask>",
    "bos_token": "<bos>",
    "eos_token": "<eos>",
    "cls_token": "<cls>",
}

MAX_LENGTH = 4096
SHARDS = 16


def gig(b: int) -> str:
    return f"{round(b / (1024 ** 3), 2)}G"


def mem() -> int:
    return psutil.Process(os.getpid()).memory_info().rss


class DatasetGen:
    def __init__(self, num_files: Optional[int] = None) -> None:
        self.files = sorted(list(DATA_PATH.iterdir()))
        if num_files is not None:
            self.files = self.files[0:num_files]
        self.keys = pd.read_csv(LABELS_PATH, index_col=0).to_dict()["Class"]
        self.iteration = 0

    def __call__(self):
        return iter(self)

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.files)

    def __next__(self) -> dict[str, str | int]:
        if self.iteration >= len(self):
            raise StopIteration

        f = self.files[self.iteration]
        with open(f, encoding="utf-8") as handle:
            s = handle.read()

        l = self.keys.get(f.stem, None)
        self.iteration += 1

        return {"text": s, "label": l, "file": f.as_posix()}


def main(mode: str, num_files: int, batch_size: int, writer_batch_size: int, num_proc: int):
    print(f"RSS INITIAL: {gig(mem())}")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH.as_posix())
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        model_max_length=MAX_LENGTH,
    )
    tokenizer.add_special_tokens(SPECIALS)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], max_length=MAX_LENGTH, truncation=False)

    dataset = Dataset.from_generator(DatasetGen(num_files))
    print(f"DATASET SIZE: {gig(dataset.dataset_size)}")
    print(f"RSS AFTER LOADING: {gig(mem())}")

    if mode == "BASIC":
        print("You can watch the memory usage increase with the top command.")
        dataset = dataset.map(
            tokenize_fn,
            batched=True,
            batch_size=batch_size,
            writer_batch_size=writer_batch_size,
            num_proc=num_proc,
        )
        return

    if mode == "SHARD":
        shards = [dataset.shard(SHARDS, i) for i in range(SHARDS)]
        print(f"RSS AFTER SHARDING: {gig(mem())}")
        for i in range(SHARDS):
            shards[i] = shards[i].map(
                tokenize_fn,
                batched=True,
                batch_size=batch_size,
                writer_batch_size=writer_batch_size,
                num_proc=num_proc,
            )
            print(f"RSS AFTER SHARDING ITER {i}: {gig(mem())}")
        dataset = concatenate_datasets(shards)
        return


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["BASIC", "SHARD"])
    parser.add_argument("--num_files", type=int, default=673)
    parser.add_argument("--batch_size", type=int, default=1000)
    parser.add_argument("--writer_batch_size", type=int, default=1000)
    parser.add_argument("--num_proc", type=int, default=1)
    args = parser.parse_args()
    main(args.mode, args.num_files, args.batch_size, args.writer_batch_size, args.num_proc)

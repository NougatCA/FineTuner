import os
from dataclasses import dataclass
import json


@dataclass
class ClassificationExample:
    """Raw example of classification tasks, source_pair is used for input-pair classification."""
    idx: str
    source: str
    label: int
    source_pair: str = None


@dataclass
class RetrievalExample:
    """Raw example of retrieval tasks."""
    idx: str
    query: str
    match: str = None


@dataclass
class Seq2SeqExample:
    """Raw example of seq2seq tasks, is also of all T5-based models."""
    idx: str
    source: str
    target: str


def load_examples(args, split, aux_file=None):

    data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    if args.subset:
        data_dir = os.path.join(args.data_dir, args.subset)

    examples = []
    if args.task == "defect":
        assert split in ["train", "valid", "test"]
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            for _, line in enumerate(f):
                line = line.strip()
                js = json.loads(line)
                code = ' '.join(js['func'].split())
                examples.append(
                    ClassificationExample(
                        idx=js['idx'],
                        source=code,
                        label=int(js['target'])
                    )
                )
    elif args.task == "clone":
        assert split in ["train", "valid", "test"]
        with open(os.path.join(data_dir, f"{split}.txt"), mode="r", encoding="utf-8") as f:
            for

import os
from dataclasses import dataclass
import json
from tqdm import tqdm


@dataclass
class ClassificationExample:
    """Raw example of classification tasks."""
    idx: str
    source: str
    label: int
    # used for input-pair classification
    source_pair: str = None


@dataclass
class RetrievalExample:
    """Raw example of retrieval tasks."""
    idx: str
    source: str
    label: str


@dataclass
class SearchExample:
    """Raw example for search tasks."""
    idx: str
    query: str
    match: str = None


@dataclass
class Seq2SeqExample:
    """Raw example of seq2seq tasks, is also of all T5-based models."""
    idx: str
    source: str
    target: str


def load_aux_data(args):

    data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    if args.subset:
        data_dir = os.path.join(args.data_dir, args.subset)

    # bigclonebench
    if args.task == "clone":
        if args.dataset == "bigclonebench":
            idx_to_code = {}
            with open(os.path.join(data_dir, "data.jsonl"), mode="r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in tqdm(lines, total=len(lines), desc="Loading raw data"):
                    js = json.loads(line.strip())
                    code = ' '.join(js['func'].split())
                    idx_to_code[js['idx']] = code
            return idx_to_code
    elif args.task == "exception":
        with open(os.path.join(data_dir, "types.txt"), mode="r", encoding="utf-8") as f:
            type_list = f.read().split()
        return type_list
    return None


def load_examples(args, split, aux_data=None):

    data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    if args.subset:
        data_dir = os.path.join(args.data_dir, args.subset)

    examples = []
    if args.task == "defect":
        assert split in ["train", "valid", "test"]
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines), desc=f"Loading {split} set"):
                line = line.strip()
                js = json.loads(line)
                code = ' '.join(js['func'].split())
                examples.append(ClassificationExample(idx=js['idx'],
                                                      source=code,
                                                      label=int(js['target'])))
    elif args.task == "clone":
        assert split in ["train", "valid", "test"]
        assert aux_data     # a dict map from idx to code
        with open(os.path.join(data_dir, f"{split}.txt"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(tqdm(lines, total=len(lines), desc=f"Loading {split} set")):
                url1, url2, label = line.strip().split('\t')
                if url1 not in aux_data or url2 not in aux_data:
                    continue
                label = int(label)
                examples.append(ClassificationExample(idx=str(idx),
                                                      source=aux_data[url1],
                                                      source_pair=aux_data[url2],
                                                      label=label))
    elif args.task == "exception":
        assert split in ["train", "valid", "test"]
        assert aux_data     # a list of all types
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for idx, line in enumerate(tqdm(lines, total=len(lines), desc=f"Loading {split} set")):
                js = json.loads(line.strip())
                code = " ".join(js["function"].split())
                target_txt = js["label"].lower()
                examples.append(ClassificationExample(idx=str(idx),
                                                      source=code,
                                                      label=aux_data.index(target_txt)))

    elif args.task == "retrieval":
        assert split in ["train", "valid", "test"]
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
            for line in tqdm(lines, total=len(lines), desc=f"Loading {split} set"):
                line = line.strip()
                js = json.loads(line)
                code = " ".join(js["code"].split())
                examples.append(RetrievalExample(idx=js["index"],
                                                 source=code,
                                                 label=js["label"]))

    elif args.task == "search":
        pass

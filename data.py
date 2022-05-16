
import logging
import os
import random
from dataclasses import dataclass
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)


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
    query: str = None
    url: str = None
    code: str = None


@dataclass
class CoSQAExample:
    """Raw example for CoSQA tasks."""
    idx: str
    code: str
    nl: str
    label: int


@dataclass
class Seq2SeqExample:
    """Raw example of seq2seq tasks, is also of all T5-based models."""
    idx: str
    source: str
    target: str
    metadata: dict = None


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
    # exception
    elif args.task == "exception":
        with open(os.path.join(data_dir, "types.txt"), mode="r", encoding="utf-8") as f:
            type_list = f.read().split()
        return type_list
    return None


def load_examples(args, split, aux_data=None):
    assert split in ["train", "valid", "test"]

    data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    if args.subset and args.task != "translation":
        data_dir = os.path.join(args.data_dir, args.subset)

    examples = []
    if args.task == "defect":
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc=f"Loading {split} data"):
            line = line.strip()
            js = json.loads(line)
            code = ' '.join(js['func'].split())
            examples.append(ClassificationExample(idx=js['idx'],
                                                  source=code,
                                                  label=int(js['target'])))
    elif args.task == "clone":
        assert aux_data     # a dict map from idx to code
        with open(os.path.join(data_dir, f"{split}.txt"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, total=len(lines), desc=f"Loading {split} data")):
            url1, url2, label = line.strip().split('\t')
            if url1 not in aux_data or url2 not in aux_data:
                continue
            label = int(label)
            examples.append(ClassificationExample(idx=str(idx),
                                                  source=aux_data[url1],
                                                  source_pair=aux_data[url2],
                                                  label=label))
    elif args.task == "exception":
        assert aux_data     # a list of all types
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, total=len(lines), desc=f"Loading {split} data")):
            js = json.loads(line.strip())
            code = " ".join(js["function"].split())
            target_txt = js["label"].lower()
            examples.append(ClassificationExample(idx=str(idx),
                                                  source=code,
                                                  label=aux_data.index(target_txt)))

    elif args.task == "retrieval":
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc=f"Loading {split} data"):
            js = json.loads(line.strip())
            code = " ".join(js["code"].split())
            examples.append(RetrievalExample(idx=js["index"],
                                             source=code,
                                             label=js["label"]))

    elif args.task == "search":
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc=f"Loading {split} data"):
            js = json.loads(line.strip())
            if 'code_tokens' in js:
                source = ' '.join(js['code_tokens'])
            else:
                source = ' '.join(js['function_tokens'])
            doc = ' '.join(js['docstring_tokens'])
            examples.append(SearchExample(idx=js["idx"],
                                          query=doc,
                                          code=source,
                                          url=js["url"]))

    elif args.task == "cosqa":
        test_answers = {}
        if split == "test":
            with open(os.path.join(data_dir, "answers.txt"), mode="r", encoding="utf-8") as answer_f:
                for line in answer_f:
                    idx, label = line.strip().split("\t")
                    test_answers[idx] = int(label)
        with open(os.path.join(data_dir, "test_webquery" if split == "test" else f"cosqa-{split}.json"),
                  mode="r", encoding="utf-8") as f:
            data = json.load(f)
        for js in tqdm(data, total=len(data), desc=f"Loading {split} data"):
            code = " ".join(js["code"])
            nl = " ".join(js["doc"])
            idx = js["idx"]
            if split == "test":
                label = test_answers[idx]
            else:
                label = js["label"]
            examples.append(CoSQAExample(idx=idx,
                                         code=code,
                                         nl=nl,
                                         label=label))
    elif args.task in ["translation", "fixing", "mutant", "assert"]:
        if args.task == "translation":
            source_lang, target_lang = args.subset.split("-")
            source_path = os.path.join(data_dir, f"{split}.java-cs.txt.{source_lang}")
            target_path = os.path.join(data_dir, f"{split}.java-cs.txt.{target_lang}")
        elif args.task == "fixing":
            source_path = os.path.join(data_dir, f"{split}.buggy-fixed.buggy")
            target_path = os.path.join(data_dir, f"{split}.buggy-fixed.fixed")
        elif args.task == "mutant":
            source_path = os.path.join(data_dir, f"{split}.fixed.txt")
            target_path = os.path.join(data_dir, f"{split}.buggy.txt")
        elif args.task == "assert":
            source_path = os.path.join(data_dir, f"{split}_methods.txt")
            target_path = os.path.join(data_dir, f"{split}_assert.txt")
        else:
            raise ValueError(f"The task {args.task} is not supported.")

        with open(source_path, mode="r", encoding="utf-8") as source_f, \
             open(target_path, mode="r", encoding="utf-8") as target_f:
            source_lines = source_f.readlines()
            target_lines = target_f.readlines()
        assert len(source_lines) == len(target_lines)
        for idx, (source_line, target_line) in enumerate(tqdm(zip(source_lines, target_lines),
                                                              total=len(source_lines),
                                                              desc=f"Loading {split} data")):
            examples.append(Seq2SeqExample(idx=str(idx),
                                           source=source_line.strip(),
                                           target=target_line.strip()))

    elif args.task == "completion":
        pass

    elif args.task in ["summarization", "generation"]:
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for idx, line in enumerate(tqdm(lines, total=len(lines), desc=f"Loading {split} data")):
            js = json.loads(line.strip())
            if args.task == "summarization":
                source = " ".join(js["code_tokens"])
                target = " ".join(js["docstring_tokens"])
            else:
                source = " ".join(js["nl"])
                target = " ".join(js["code"])
            examples.append(Seq2SeqExample(idx=str(idx),
                                           source=source,
                                           target=target))
    logger.info(f"{split} data loaded, total size: {len(examples)}")

    if args.training_sample > 0:
        if args.training_sample < 1:
            num_to_sample = int(len(examples) * args.training_sample)
            examples = random.sample(examples, num_to_sample)
        elif args.training_sample >= 1:
            examples = random.sample(examples, args.training_sample)
        logger.info(f"Sampled {len(examples)} data because `--training-sample={args.training_sample}`")

    return examples


def get_model_input_ids(args, source, source_pair=None):
    pass


def encode_classification_examples(example):
    pass


def encode_retrieval_examples(example):
    pass


def encode_search_examples(example):
    pass


def encode_cosqa_example(example):
    pass


def encode_seq2seq_example(example):
    pass


def create_dataset(args, examples, tokenizer, split):


    if args.task in ["defect", "clone", "exception"]:
        pass

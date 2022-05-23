import torch
from torch.utils.data.dataset import Dataset, TensorDataset
from torch.utils.data.dataloader import DataLoader
import logging
import os
import random
from dataclasses import dataclass
import json
from functools import partial
from typing import List, Union, Tuple
from tqdm import tqdm
import multiprocessing

import configs

logger = logging.getLogger(__name__)


@dataclass
class ClassificationExample:
    """Raw example of classification tasks."""
    idx: str
    source: str
    label: int
    # used for input-pair classification
    source_pair: str = None
    # used for t5-based models
    label_txt: str = None


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
    url: str = None
    code: str = None
    nl: str = None


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
    """Loads auxiliary datafile besides the train/valid/test data file."""
    data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    if args.subset:
        data_dir = os.path.join(args.data_dir, args.subset)

    # BigCloneBench
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


def load_examples(args, split, aux_data=None) -> List:
    """Loads raw examples from the train/valid/test file."""
    assert split in ["train", "valid", "test"]

    data_dir = os.path.join(args.data_dir, args.task, args.dataset)
    if args.subset and args.task != "translation":
        data_dir = os.path.join(data_dir, args.subset)

    logger.info(f"Start loading {split} data from {data_dir}")

    examples = []
    if args.task == "defect":
        with open(os.path.join(data_dir, f"{split}.jsonl"), mode="r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in tqdm(lines, total=len(lines), desc=f"Loading {split} data"):
            js = json.loads(line.strip())
            code = " ".join(js["func"].split())
            label = int(js["target"])
            examples.append(ClassificationExample(idx=js["idx"],
                                                  source=code,
                                                  label=label,
                                                  label_txt="true" if label == 1 else "false"))
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
                                                  label=label,
                                                  label_txt="true" if label == 1 else "false"))
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
                                                  label=aux_data.index(target_txt),
                                                  label_txt=target_txt))

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
                                          nl=doc,
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

    # sample specific number/ratio of examples if needed
    if args.training_sample is not None and args.training_sample > 0:
        if args.training_sample < 1:
            num_to_sample = int(len(examples) * args.training_sample)
            examples = random.sample(examples, num_to_sample)
        elif args.training_sample >= 1:
            examples = random.sample(examples, args.training_sample)
        logger.info(f"Sampled {len(examples)} data because `--training-sample={args.training_sample}`")

    return examples


@dataclass
class ClassificationInputFeature:
    input_ids: List[int]
    label: int = None


@dataclass
class Seq2SeqInputFeature:
    input_ids: List[int]
    decoder_input_ids: list = None


@dataclass
class RetrievalInputFeature:
    input_ids: List[int]
    idx: int
    label: int


@dataclass
class SearchInputFeature:
    code_ids: List[int]
    nl_ids: List[int]
    url: str
    idx: str


@dataclass
class CoSQAInputFeature:
    code_ids: List[int]
    nl_ids: List[int]
    idx: str
    label: int


class RetrievalDataset(Dataset):

    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        label = self.features[i].label
        idx = self.features[i].idx

        pos_feature = None
        neg_feature = None
        while True:
            random_feature = random.sample(self.features, 1)[0]
            if pos_feature is None and random_feature.idx == idx:
                pos_feature = random_feature
            if neg_feature is None and random_feature.idx != idx:
                neg_feature = random_feature
            if pos_feature is not None and neg_feature is not None:
                break

        return torch.tensor(self.features[i].input_ids), torch.tensor(pos_feature.input_ids), \
            torch.tensor(neg_feature.input_ids), torch.tensor(label)


def remove_special_tokens(s, tokenizer) -> str:
    """Removes all special tokens from given str."""
    for token in tokenizer.all_special_tokens:
        s = s.replace(token, " ")
    return s


def build_model_input_ids(source, tokenizer, max_length):
    """Builds model-specific input."""
    source = remove_special_tokens(source, tokenizer)
    return tokenizer.encode(source, padding="max_length", max_length=max_length, truncation=True)


def convert_code_to_input_ids(source, tokenizer, max_length, source_pair=None) -> List[int]:
    """Converts code to input ids, only for code."""
    input_ids = build_model_input_ids(source=source, tokenizer=tokenizer, max_length=max_length)
    if source_pair:
        input_pair_ids = build_model_input_ids(source=source_pair, tokenizer=tokenizer, max_length=max_length)
        input_ids += input_pair_ids
    return input_ids


def encode_classification_example(example: ClassificationExample, tokenizer, max_length) -> ClassificationInputFeature:
    input_ids = convert_code_to_input_ids(source=example.source,
                                          tokenizer=tokenizer,
                                          max_length=max_length,
                                          source_pair=example.source_pair)
    return ClassificationInputFeature(input_ids=input_ids, label=example.label)


def encode_t5_classification_example(example: ClassificationExample, tokenizer, max_source_length, max_target_length) \
        -> Seq2SeqInputFeature:
    input_ids = convert_code_to_input_ids(source=example.source,
                                          tokenizer=tokenizer,
                                          max_length=max_source_length,
                                          source_pair=example.source_pair)
    decoder_input_ids = tokenizer.encode(example.label_txt,
                                         padding="max_length",
                                         max_length=max_target_length,
                                         truncation=True)
    return Seq2SeqInputFeature(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


def encode_retrieval_example(example: RetrievalExample, tokenizer, max_length) -> RetrievalInputFeature:
    input_ids = convert_code_to_input_ids(source=example.source,
                                          tokenizer=tokenizer,
                                          max_length=max_length)
    return RetrievalInputFeature(input_ids=input_ids,
                                 idx=int(example.idx),
                                 label=int(example.label))


def encode_search_example(example: SearchExample, tokenizer, max_length) -> SearchInputFeature:
    code_ids = convert_code_to_input_ids(source=example.code,
                                         tokenizer=tokenizer,
                                         max_length=max_length)
    nl_ids = tokenizer.encode(example.nl,
                              padding="max_length",
                              max_length=max_length,
                              truncation=True)
    return SearchInputFeature(code_ids=code_ids,
                              nl_ids=nl_ids,
                              url=example.url,
                              idx=example.idx)


def encode_cosqa_example(example: CoSQAExample, tokenizer, max_length) -> CoSQAInputFeature:
    code_ids = convert_code_to_input_ids(source=example.code, tokenizer=tokenizer, max_length=max_length)
    nl_ids = tokenizer.encode(example.nl,
                              padding="max_length",
                              max_length=max_length,
                              truncation=True)
    return CoSQAInputFeature(code_ids=code_ids,
                             nl_ids=nl_ids,
                             label=example.label,
                             idx=example.idx)


def encode_seq2seq_example(example: Seq2SeqExample, tokenizer, task, max_source_length, max_target_length) \
        -> Seq2SeqInputFeature:
    if task in ["generation"]:
        input_ids = tokenizer.encode(example.source,
                                     padding="max_length",
                                     max_length=max_source_length,
                                     truncation=True)
    else:
        input_ids = convert_code_to_input_ids(source=example.source, tokenizer=tokenizer, max_length=max_source_length)
    decoder_input_ids = tokenizer.encode(example.target,
                                         padding="max_length",
                                         max_length=max_target_length,
                                         truncation=True)
    return Seq2SeqInputFeature(input_ids=input_ids, decoder_input_ids=decoder_input_ids)


def encode_casual_example(example):
    pass


def multiprocess_encoding(encode_func, examples, num_processors=None, single_thread=False) -> List:
    """Encodes examples to input features using multiprocessing."""
    processes = num_processors if num_processors else multiprocessing.cpu_count()
    if processes > 1 and not single_thread:
        with multiprocessing.Pool(processes=processes) as p:
            features = list(p.map(encode_func, tqdm(examples, total=len(examples), desc="Encoding")))
    else:
        features = [encode_func(example) for example in tqdm(examples, total=len(examples), desc="Encoding")]
    return features


def create_dataset(args, examples, tokenizer, split) -> Union[Dataset, None]:
    """Create dataset by converting examples to input features."""

    logger.info(f"Start encoding {split} data into input features")

    dataset = None

    if args.model_type in ["t5", "codet5"] and args.task in configs.TASK_TYPE_TO_LIST["classification"]:
        encode_func = partial(encode_t5_classification_example,
                              tokenizer=tokenizer,
                              max_source_length=args.max_source_length,
                              max_target_length=args.max_target_length)
        features = multiprocess_encoding(encode_func, examples)
        all_input_ids, all_decoder_input_ids = [], []
        for f in features:
            all_input_ids.append(f.input_ids)
            all_decoder_input_ids.append(f.decoder_input_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_decoder_input_ids = torch.tensor(all_decoder_input_ids, dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_decoder_input_ids)

    elif args.task in configs.TASK_TYPE_TO_LIST["classification"]:
        encode_func = partial(encode_classification_example, tokenizer=tokenizer, max_length=args.max_source_length)
        features = multiprocess_encoding(encode_func, examples)
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_labels = torch.tensor([[f.label] for f in features], dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_labels)

    elif args.task == "retrieval":
        encode_func = partial(encode_retrieval_example, tokenizer=tokenizer, max_length=args.max_source_length)
        features = multiprocess_encoding(encode_func, examples)
        dataset = RetrievalDataset(features)

    elif args.task == "search":
        encode_func = partial(encode_search_example, tokenizer=tokenizer, max_length=args.max_source_length)
        features = multiprocess_encoding(encode_func, examples)
        all_code_ids, all_nl_ids = [], []
        for f in features:
            all_code_ids.append(f.code_input_ids)
            all_nl_ids.append(f.nl_input_ids)
        all_code_ids = torch.tensor(all_code_ids, dtype=torch.long)
        all_nl_ids = torch.tensor(all_nl_ids, dtype=torch.long)
        dataset = TensorDataset(all_code_ids, all_nl_ids)

    elif args.task == "cosqa":
        encode_func = partial(encode_cosqa_example, tokenizer=tokenizer, max_length=args.max_source_length)
        features = multiprocess_encoding(encode_func, examples)
        all_code_ids, all_nl_ids, all_labels = [], [], []
        for f in features:
            all_code_ids.append(f.code_input_ids)
            all_nl_ids.append(f.nl_input_ids)
            all_labels.append([f.label])
        all_code_ids = torch.tensor(all_code_ids, dtype=torch.long)
        all_nl_ids = torch.tensor(all_nl_ids, dtype=torch.long)
        all_labels = torch.tensor(all_labels, dtype=torch.long)
        dataset = TensorDataset(all_code_ids, all_nl_ids, all_labels)

    elif args.task in configs.TASK_TYPE_TO_LIST["seq2seq"]:
        encode_func = partial(encode_seq2seq_example,
                              tokenizer=tokenizer,
                              task=args.task,
                              max_source_length=args.max_source_length,
                              max_target_length=args.max_target_length)
        features = multiprocess_encoding(encode_func, examples)
        all_input_ids, all_decoder_input_ids = [], []
        for f in features:
            all_input_ids.append(f.input_ids)
            all_decoder_input_ids.append(f.decoder_input_ids)
        all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
        all_decoder_input_ids = torch.tensor(all_decoder_input_ids, dtype=torch.long)
        dataset = TensorDataset(all_input_ids, all_decoder_input_ids)

    elif args.task == "completion":
        dataset = None

    logger.info(f"{split} data encoded, total size: {len(dataset)}")

    return dataset


def prepare_data(args, split, tokenizer) -> Tuple[List, Dataset, DataLoader]:
    """Prepares data-related instances, such as raw examples, dataset and dataloader."""
    aux_data = None
    if args.task in ["exception"] or args.dataset in ["bigclonebench"]:
        aux_data = load_aux_data(args)

    examples = load_examples(args, split=split, aux_data=aux_data)
    dataset = create_dataset(args, examples=examples, tokenizer=tokenizer, split=split)
    dataloader = DataLoader(dataset,
                            shuffle=True if split == "train" else False,
                            batch_size=args.train_batch_size if split == "train" else args.eval_batch_size,
                            num_workers=4,
                            pin_memory=True)

    return examples, dataset, dataloader

import math
import pickle
from functools import reduce

import numpy as np
import torch
from bricks.parser import (DFG_csharp, DFG_go, DFG_java, DFG_javascript,
                           DFG_php, DFG_python, DFG_ruby)
from torch.utils.data.dataset import Dataset
from tree_sitter import Language, Parser

from syncobert.model import init_model

dfg_function = {
    'python': DFG_python,
    'java': DFG_java,
    'go': DFG_go,
    'php': DFG_php,
    'ruby': DFG_ruby,
    'c_sharp': DFG_csharp,
    'javascript': DFG_javascript
}

parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('parser/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser


def dfs(cursor, hook):
    hook(cursor)
    if cursor.goto_first_child():
        dfs(cursor, hook)
        cursor.goto_parent()

    while cursor.goto_next_sibling():
        dfs(cursor, hook)


def remove_diagonal(mat):
    l = len(mat)
    mat = mat.flatten()[:-1].reshape(l - 1, l + 1)
    mat = mat[:, 1:]
    mat = mat.flatten().reshape(l, l - 1)
    return mat


def padding(l, max_len, pad_token_id):
    return l + [pad_token_id] * (max_len - len(l))


def parse(source, lang, nl, tokenizer, masker, scene='first'):

    nl_encoding = tokenizer(nl, max_length=510, truncation=True)

    code_encoding = tokenizer(source, max_length=510, truncation=True)
    code_tokens = code_encoding.tokens()

    tree = parsers[lang][0].parse(bytes(source, 'utf8'))

    identi_dict = set()

    def hook(cursor):
        if cursor.node.type == 'identifier':
            identi_dict.add((cursor.node.start_byte, cursor.node.end_byte))

    dfs(tree.walk(), hook)
    is_identi = [False] * len(code_tokens)
    for i in range(1, len(code_tokens) - 1):
        start, end = code_encoding.token_to_chars(i)

        for (s, e) in identi_dict:
            if e >= start >= s or s <= end <= e:
                is_identi[i] = True

    def dfs2(cursor, stack, ast_string):
        start, end = cursor.node.start_byte, cursor.node.end_byte

        if cursor.node.type == 'module':
            pass
        elif len(cursor.node.children) == 0:
            ast_string.append(
                (source[start:end], stack[-1] if len(stack) > 0 else -1))
        else:
            ast_string.append(
                (cursor.node.type, stack[-1] if len(stack) > 0 else -1))

        if cursor.goto_first_child():
            stack.append(len(ast_string) - 1)
            dfs2(cursor, stack, ast_string)
            stack.pop()
            cursor.goto_parent()

        while cursor.goto_next_sibling():
            dfs2(cursor, stack, ast_string)

    ast_string = []
    stack = []
    dfs2(tree.walk(), stack, ast_string)

    def add_blank(l):
        nl = []
        for i, t in enumerate(l):
            if t[1] != -1:
                nl.append((t[0], t[1] * 2))
            else:
                nl.append(t)
            if i != len(l) - 1:
                nl.append((" ", -1))
        return nl

    ast_string = add_blank(ast_string)

    ast_encoding = tokenizer(
        "".join([t for t, _, in ast_string]), max_length=510, truncation=True)

    edge_mat = np.zeros((256, 256))
    ast2token = {}
    for i in range(1, len(ast_encoding.tokens()) - 1):
        start, end = ast_encoding.token_to_chars(i)
        amount = 0
        for j in range(len(ast_string)):
            new_amount = amount + len(ast_string[j][0])
            if ast_string[j][0] != " " and (amount <= start <= new_amount or amount <= end <= new_amount):
                if j in ast2token:
                    ast2token[j].append(i)
                else:
                    ast2token[j] = [i]
                if ast_string[j][1] != -1:
                    for k in ast2token[ast_string[j][1]]:
                        if k != -1:

                            if i + 2 < 256 and k + 2 < 256:
                                edge_mat[i + 2][k +
                                                2] = edge_mat[k + 2][i + 2] = 1
            amount = new_amount

    if scene == 'first':

        nl_ids = [tokenizer.cls_token_id] + nl_encoding.input_ids[:95]
        code_ids = [tokenizer.sep_token_id] + code_encoding.input_ids[:159]
        ast_ids = [tokenizer.sep_token_id] + ast_encoding.input_ids[:255]

        origin_ids = nl_ids + code_ids + ast_ids
        origin_ids = padding(origin_ids, 512, tokenizer.pad_token_id)
        origin_ids = torch.tensor(origin_ids)

        masked_ids, labels = masker.mask_seq(origin_ids)

        code_pos = len(nl_ids)
        code_pos = torch.tensor(code_pos)

        ast_pos = len(nl_ids) + len(code_ids)
        ast_pos = torch.tensor(ast_pos)

        edge_mat = torch.tensor(edge_mat)

        is_identi = [0] + is_identi
        is_identi = is_identi[:160] if len(
            is_identi) >= 160 else is_identi + [0] * (160 - len(is_identi))
        is_identi = torch.tensor(is_identi)

        ast_sect = (ast_pos, ast_pos + len(ast_ids))
        attention_mask = atten_mask(origin_ids, ast_sect, edge_mat)
        torch.tensor(attention_mask)

        return masked_ids, labels, code_pos, ast_pos, edge_mat, is_identi, attention_mask

    elif scene == 'second':
        nl_ids = [tokenizer.cls_token_id] + nl_encoding.input_ids[:511]
        nl_ids = padding(nl_ids, 512, tokenizer.pad_token_id)
        nl_ids = torch.tensor(nl_ids)
        masked_nl_ids, _, = masker.mask_seq(nl_ids)

        code_ids = [tokenizer.cls_token_id] + code_encoding.input_ids[:255] + [
            tokenizer.sep_token_id] + ast_encoding.input_ids[:255]
        code_ids = padding(code_ids, 512, tokenizer.pad_token_id)
        code_ids = torch.tensor(code_ids)
        masked_code_ids, _, = masker.mask_seq(code_ids)
        return masked_nl_ids, masked_code_ids
    elif scene == 'third':
        first_ids = [tokenizer.cls_token_id] + nl_encoding.input_ids[:95] + \
                    [tokenizer.sep_token_id] + code_encoding.input_ids[:159] + \
                    [tokenizer.sep_token_id] + ast_encoding.input_ids[:255]
        first_ids = padding(first_ids, 512, tokenizer.pad_token_id)
        first_ids = torch.tensor(first_ids)
        masked_first_ids, _, = masker.mask_seq(first_ids)

        second_ids = [tokenizer.cls_token_id] + nl_encoding.input_ids[:95] + \
                     [tokenizer.sep_token_id] + ast_encoding.input_ids[:255] + \
                     [tokenizer.sep_token_id] + code_encoding.input_ids[:159]
        second_ids = padding(second_ids, 512, tokenizer.pad_token_id)
        second_ids = torch.tensor(second_ids)
        masked_second_ids, _, = masker.mask_seq(second_ids)
        return masked_first_ids, masked_second_ids


def atten_mask(seq, ast_sect, edge_mat):
    assert ast_sect[1] - ast_sect[0] <= len(edge_mat)
    max_len = len(seq)
    attn_mask = np.ones((max_len, max_len), dtype=bool)
    for i in range(max_len):
        if seq[i] == 0:
            attn_mask[i, :] = False
            attn_mask[:, i] = False

        if i >= ast_sect[0] and i < ast_sect[1]:
            for j in range(len(edge_mat)):
                if edge_mat[i - ast_sect[0]][j] == 1:
                    attn_mask[i, ast_sect[0] + j] = False
                    attn_mask[ast_sect[0] + j, i] = False

    return attn_mask


class Masker:
    def __init__(
            self,
            mask_prob=0.15,
            replace_prob=0.9,
            num_tokens=None,
            random_token_prob=0.,
            mask_token_id=2,
            pad_token_id=0,
            mask_ignore_token_ids=[]):

        self.mask_prob = mask_prob
        self.replace_prob = replace_prob

        self.num_tokens = num_tokens
        self.random_token_prob = random_token_prob

        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set(
            [*mask_ignore_token_ids, pad_token_id])

    def mask_seq(self, seq):
        one = False
        if len(seq.shape) == 1:
            seq = seq.unsqueeze(0)
            one = True

        no_mask = self.mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = self.get_mask_subset_with_prob(~no_mask, self.mask_prob)

        masked_seq = seq.clone().detach()

        labels = seq.masked_fill(~mask, self.pad_token_id)

        if self.random_token_prob > 0:
            assert self.num_tokens is not None, 'num_tokens keyword must be supplied when instantiating MLM if using random token replacement'
            random_token_prob = self.prob_mask_like(
                seq, self.random_token_prob)
            random_tokens = torch.randint(
                0, self.num_tokens, seq.shape, device=seq.device)
            random_no_mask = self.mask_with_tokens(
                random_tokens, self.mask_ignore_token_ids)
            random_token_prob &= ~random_no_mask
            masked_seq = torch.where(
                random_token_prob, random_tokens, masked_seq)

            mask = mask & ~random_token_prob

        replace_prob = self.prob_mask_like(seq, self.replace_prob)
        masked_seq = masked_seq.masked_fill(
            mask * replace_prob, self.mask_token_id)

        if one:
            return masked_seq[0], labels[0]
        return masked_seq, labels

    def prob_mask_like(self, t, prob):
        return torch.zeros_like(t).float().uniform_(0, 1) < prob

    def mask_with_tokens(self, t, token_ids):
        init_no_mask = torch.full_like(t, False, dtype=torch.bool)
        mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
        return mask

    def get_mask_subset_with_prob(self, mask, prob):
        batch, seq_len, device = *mask.shape, mask.device
        max_masked = math.ceil(prob * seq_len)

        num_tokens = mask.sum(dim=-1, keepdim=True)
        mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
        mask_excess = mask_excess[:, :max_masked]

        rand = torch.rand((batch, seq_len),
                          device=device).masked_fill(~mask, -1e9)
        _, sampled_indices = rand.topk(max_masked, dim=-1)
        sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)

        new_mask = torch.zeros((batch, seq_len + 1), device=device)
        new_mask.scatter_(-1, sampled_indices, 1)
        return new_mask[:, 1:].bool()


class PretrainDataset(Dataset):
    def __init__(self, tokenizer):
        super(PretrainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.masker = Masker(mask_prob=0.15,
                             replace_prob=0.9,
                             num_tokens=None,
                             random_token_prob=0.,
                             mask_token_id=2,
                             pad_token_id=0,
                             mask_ignore_token_ids=tokenizer.convert_tokens_to_ids(tokenizer.all_special_tokens))

    def set_datas(self, another_dataset):
        self.paths, self.languages, self.sources, self.codes, self.asts, self.names, self.codes_wo_name, self.names_wo_name, self.only_names, self.docs = another_dataset.paths, another_dataset.languages, another_dataset.sources, another_dataset.codes, another_dataset.asts, another_dataset.names, another_dataset.codes_wo_name, another_dataset.names_wo_name, another_dataset.only_names, another_dataset.docs
        self.size = len(self.codes)

    def __getitem__(self, index, scene):
        lang = self.languages[index]
        code = self.codes[index]
        nl = self.docs[index]
        return parse(code, lang, nl, self.tokenizer, self.masker, scene=scene)

    def __len__(self):
        return len(self.codes)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.set_datas(obj)

        return self


class Scene1Dataset(Dataset):
    def __init__(self, dataset):
        super(Scene1Dataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index, scene='first')


class Scene2Dataset(Dataset):
    def __init__(self, dataset):
        super(Scene2Dataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index, scene='second')


class Scene3Dataset(Dataset):
    def __init__(self, dataset):
        super(Scene3Dataset, self).__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset.__getitem__(index, scene='third')


if __name__ == '__main__':
    model, tokenizer = init_model()
    dataset = PretrainDataset(tokenizer).load('/datasets/pre_train.pkl')
    print(len(dataset))
    for i in range(len(dataset)):
        dataset[i]

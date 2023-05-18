import os

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from syncobert.tasks import IP, MCL, MLM, TEP


class SyncoBert(nn.Module):

    def __init__(self, base_model):
        super(SyncoBert, self).__init__()
        self.base_model = base_model

    def forward(self, input_ids, attention_mask=None):
        if attention_mask == None:
            attention_mask = input_ids.ne(1)
        return self.base_model(input_ids, attention_mask)

    def get_all_embedding(self, input_ids, attention_mask=None):
        return self.forward(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

    def get_cls_embedding(self, input_ids, attention_mask=None):
        return self.get_all_embedding(input_ids, attention_mask)[:, 0, :]

    def get_regularization(self, λ=1e-4):

        return λ * torch.norm(torch.cat([p.view(-1) for p in self.base_model.parameters() if p.requires_grad]), 2)

    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.state_dict(), path)

    def load_state_dict(self, path):
        self.load_state_dict(torch.load(path))


class TaskModel(nn.Module):

    def __init__(self, base_model, eps, vocab_size):
        super().__init__()
        self.syncobert = SyncoBert(base_model)
        self.ip = IP()
        self.mcl = MCL()
        self.mlm = MLM(eps, vocab_size)
        self.tep = TEP()

    def forward(self, input_io, scene='first'):
        if scene == 'first':
            return self.scene1(input_io)
        elif scene == 'second':
            return self.scene2(input_io)
        elif scene == 'third':
            return self.scene3(input_io)

    def scene1(self, input_io):
        embs = self.syncobert.get_all_embedding(
            input_io['masked_ids'], input_io['attn_mask'])
        mlm_loss = self.mlm(embs, input_io['labels'])
        ip_loss = self.ip(embs, input_io['is_identi'], input_io['code_pos'])
        tep_loss = self.tep(
            embs, input_io['edge_sim_mat'], input_io['ast_pos'])
        return mlm_loss + ip_loss + tep_loss + self.syncobert.get_regularization()

    def scene2(self, input_io):
        nl_embs = self.syncobert.get_all_embedding(input_io['masked_nl_ids'])
        code_embs = self.syncobert.get_all_embedding(
            input_io['masked_code_ids'])
        mcl_loss = self.mcl(
            nl_embs, input_io['masked_nl_ids'], code_embs, input_io['masked_code_ids'])
        return mcl_loss

    def scene3(self, input_io):
        first_embs = self.syncobert.get_all_embedding(input_io['first_ids'])
        second_embs = self.syncobert.get_all_embedding(input_io['second_ids'])
        return self.mcl(first_embs, input_io['first_ids'], second_embs, input_io['second_ids'])


def init_model():
    model = AutoModel.from_pretrained("codebert-base")
    tokenizer = AutoTokenizer.from_pretrained("codebert-base")
    return model, tokenizer


if __name__ == '__main__':
    model, tokenizer = init_model()
    nl_tokens = tokenizer.tokenize("return maximum value")
    code_tokens = tokenizer.tokenize(
        "def max(a,b): if a>b: return a else return b")
    tokens = [tokenizer.cls_token] + nl_tokens + \
        [tokenizer.sep_token] + code_tokens + [tokenizer.eos_token]
    tokens_ids = tokenizer.convert_tokens_to_ids(tokens)

    context_embeddings = model(torch.tensor(tokens_ids)[
                               None, :]).last_hidden_state
    print(context_embeddings.shape)

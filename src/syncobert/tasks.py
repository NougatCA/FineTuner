import torch
import torch.nn.functional as F
from torch import nn


class RobertaLMHead(nn.Module):
    def __init__(self, eps, vocab_size):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.layer_norm = nn.LayerNorm(768, eps=eps)
        self.decoder = nn.Linear(768, vocab_size)
        self.bias = nn.Parameter(torch.zeros(vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, ):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x


class MLM(nn.Module):
    def __init__(self,
                 eps, vocab_size,
                 pad_token_id=0):
        super().__init__()
        self.mlm_head = RobertaLMHead(eps, vocab_size)
        self.pad_token_id = pad_token_id

    def forward(self, embs, labels):
        logits = self.mlm_head(embs)
        mlm_loss = F.cross_entropy(
            logits.transpose(1, 2),
            labels,
            ignore_index=self.pad_token_id,
        )
        return mlm_loss


class MCL(nn.Module):
    def __init__(self):
        super().__init__()

    def get_average(self, embs, ids):
        x = (embs * ids.ne(1)[:, :, None]).sum(1) / ids.ne(1).sum(-1)[:, None]

        x = F.normalize(x, p=2, dim=1)
        return x

    def get_cls(self, embs):
        return embs[:, 0, :]

    def forward(self, embs1, ids1, embs2, ids2):
        embd1 = self.get_average(embs1, ids1)
        embd2 = self.get_average(embs2, ids2)
        sim_mat = torch.einsum("md,nd->mn", embd1, embd2)
        bs = embs1.shape[0]
        label = torch.arange(bs, device=sim_mat.device)
        return F.cross_entropy(sim_mat * 20., label)


class ClassificationHead(nn.Module):
    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, embedding):
        x = self.dropout(embedding)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


def gather_embs_by_pos(embs, pos, length):
    pos = pos.unsqueeze(1)
    pos = pos.repeat(1, length)
    add_idx_tensor = torch.arange(0, length, device=pos.device)
    add_idx_tensor = add_idx_tensor.repeat(pos.shape[0], 1)
    pos = pos + add_idx_tensor
    pos = pos.unsqueeze(2)
    pos = pos.expand(pos.shape[0], pos.shape[1], embs.shape[-1])
    embs = torch.gather(embs, 1, pos)

    return embs


class IP(nn.Module):
    def __init__(self):
        super().__init__()

        self.ip_head = ClassificationHead(
            input_dim=768,
            inner_dim=768,
            num_classes=2,
            pooler_dropout=0.01,
        )

    def forward(self, embs, is_identi, pos):
        embs = gather_embs_by_pos(embs, pos, is_identi.shape[1])
        probs = F.softmax(self.ip_head(embs), dim=-1)
        loss = F.cross_entropy(probs.transpose(1, 2), is_identi)
        return loss


class TEP(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, embs, edge_sim_mat, pos):
        embs = gather_embs_by_pos(embs, pos, edge_sim_mat.shape[1])
        sim_mat = self.batch_sim_mat_without_diagonal(embs)
        prob = F.sigmoid(sim_mat)
        edge_sim_mat = self.remove_diagonal(edge_sim_mat)
        return F.binary_cross_entropy_with_logits(prob, edge_sim_mat)

    def remove_diagonal(self, mat):

        bs, l, _ = mat.shape
        mat = mat.flatten().view(bs, l * l)
        mat = mat[:, :-1].reshape(bs, l - 1, l + 1)
        mat = mat[:, :, 1:]
        mat = mat.flatten().reshape(bs, l, l - 1)
        return mat

    def bmm(self, a, b):
        res = torch.einsum('bil,bil->b', a, b)
        return res

    def smm(self, a, b):
        return torch.einsum('il,il->', a, b)

    def sim_mat(self, fea):

        res = torch.matmul(fea, fea.transpose(0, 1))
        return res

    def sim_mat_without_diagonal(self, fea):

        seq, emb = fea.shape
        res = self.sim_mat(fea)
        res = res.flatten()[:-1]
        res = res.view(seq - 1, seq + 1)
        res = res[:, 1:]
        res = res.flatten().view(seq, seq - 1)
        return res

    def batch_sim_mat(self, fea):

        res = torch.matmul(fea, fea.transpose(1, 2))
        return res

    def batch_sim_mat_without_diagonal(self, fea):

        batch, seq, emb = fea.shape
        res = self.batch_sim_mat(fea)
        res = res.flatten().view(batch, -1)
        res = res[:, :-1]
        res = res.view(batch, seq - 1, seq + 1)
        res = res[:, :, 1:]
        res = res.flatten().view(batch, seq, seq - 1)
        return res

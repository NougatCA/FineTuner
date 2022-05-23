import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, EncoderDecoderConfig, EncoderDecoderModel, PreTrainedModel, \
    RobertaModel, BertModel, GPT2Model, BartModel, PLBartModel, T5Model, PreTrainedTokenizer
import logging
from prettytable import PrettyTable

import configs
from utils import human_format


logger = logging.getLogger(__name__)


def get_representation_vector(args, model: torch.nn.Module, input_dict: dict, eos_token_id) -> torch.Tensor:

    if args.model_type == "bert":
        assert isinstance(model, BertModel)
        outputs = model(**input_dict)
        return outputs[1]
    elif args.model_type == "roberta":
        assert isinstance(model, RobertaModel)
        outputs = model(**input_dict)
        return outputs[0][:, 0, :]
    elif args.model_type == "gpt2":
        assert isinstance(model, GPT2Model)
        outputs = model(**input_dict)
        seq_lens = torch.ne(input_dict["input_ids"], 0).sum(-1) - 1
        batch_size = input_dict["input_ids"].size(0)
        return outputs[0][torch.arange(batch_size), seq_lens, :]
    elif args.model_type in ["bart", "plbart", "t5", "codet5"]:
        assert isinstance(model, BartModel) or isinstance(model, PLBartModel) or isinstance(model, T5Model)
        outputs = model(**input_dict)
        hidden_states = outputs[0]  # last hidden state
        eos_mask = input_dict["input_ids"].eq(eos_token_id)
        if len(torch.unique_consecutive(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = hidden_states[eos_mask, :].view(hidden_states.size(0), -1, hidden_states.size(-1))[
                                  :, -1, :
                                  ]
        return sentence_representation
    else:
        raise ValueError(f"Model type {args.model_type} not supported.")


def prepare_input_dict_for_representation(input_ids: torch.Tensor, model_type, pad_token_id: int):
    """
    Prepares an input dict for tasks that requires getting the input representation vector,
    such as retrieval and search.
    """
    attention_mask = input_ids.ne(pad_token_id)
    input_dict = {"input_ids": input_ids,
                  "attention_mask": attention_mask}
    if model_type in ["bart", "plbart", "t5", "codet5"]:
        input_dict["labels"] = input_ids
        input_dict["decoder_attention_mask"] = attention_mask
        input_dict["output_hidden_states"] = True
    return input_dict


# class RobertaClassificationHead(nn.Module):
#     """Head for sentence-level classification tasks."""
#
#     def __init__(self, config):
#         super().__init__()
#         self.dense = nn.Linear(config.hidden_size, config.hidden_size)
#         classifier_dropout = (
#             config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
#         )
#         self.dropout = nn.Dropout(classifier_dropout)
#         self.out_proj = nn.Linear(config.hidden_size, config.num_labels)
#
#     def forward(self, features, **kwargs):
#         x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
#         x = self.dropout(x)
#         x = self.dense(x)
#         x = torch.tanh(x)
#         x = self.dropout(x)
#         x = self.out_proj(x)
#         return x


class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

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

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states


class RetrievalModel(PreTrainedModel):

    def __int__(self, args, model, tokenizer):
        super(RetrievalModel, self).__int__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, input_ids, pos_input_ids, neg_input_ids, labels=None):

        batch_size = input_ids.size(0)
        input_ids = torch.cat([input_ids, pos_input_ids, neg_input_ids], 0)
        # build input dict
        input_dict = prepare_input_dict_for_representation(input_ids,
                                                           model_type=self.args.model_type,
                                                           pad_token_id=self.tokenizer.pad_token_id)
        # get the model-specific representation vector
        vectors = get_representation_vector(self.args,
                                            model=self.model,
                                            input_dict=input_dict,
                                            eos_token_id=self.tokenizer.eos_token_id)
        # split the vector into three vectors of inputs
        outputs = vectors.split(batch_size, 0)
        vec, pos_vec, neg_vec = outputs[0], outputs[1], outputs[2]

        # distance between current example and positive example
        pos_prob = (vec * pos_vec).sum(-1)
        # distance between current example and negative example
        neg_prob = (vec * neg_vec).sum(-1)
        temp = torch.cat([vec, pos_vec], 0)
        temp_labels = torch.cat([labels, labels], 0)
        prob_3 = torch.mm(vec, temp.t())
        mask = labels[:, None] == temp_labels[None, :]
        prob_3 = prob_3 * (1 - mask.float()) - 1e9 * mask.float()

        prob = torch.softmax(torch.cat((pos_prob[:, None], neg_prob[:, None], prob_3), -1), -1)
        loss = torch.log(prob[:, 0] + 1e-10)
        loss = -loss.mean()

        return {"loss": loss, "representation_vectors": vec}


class SearchModel(PreTrainedModel):

    def __int__(self, args, model, tokenizer):
        super(SearchModel, self).__int__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.loss_fct = torch.nn.CrossEntropyLoss()

    def forward(self, code_input_ids, nl_input_ids, return_vectors=False):
        batch_size = code_input_ids.size(0)
        input_ids = torch.cat([code_input_ids, nl_input_ids], dim=0)
        # build input dict
        input_dict = prepare_input_dict_for_representation(input_ids,
                                                           model_type=self.args.model_type,
                                                           pad_token_id=self.tokenizer.pad_token_id)
        # get the model-specific representation vector
        vectors = get_representation_vector(self.args,
                                            model=self.model,
                                            input_dict=input_dict,
                                            eos_token_id=self.tokenizer.eos_token_id)

        code_vec = vectors[:batch_size]
        nl_vec = vectors[batch_size:]

        if return_vectors:
            return code_vec, nl_vec

        scores = (nl_vec[:, None, :] * code_vec[None, :, :]).sum(-1)
        loss = self.loss_fct(scores, torch.arange(batch_size, device=scores.device))
        return {"loss": loss, "code_vectors": code_vec, "nl_vectors": nl_vec}


class CoSQAModel(PreTrainedModel):

    def __int__(self, args, model, tokenizer):
        super(CoSQAModel, self).__int__()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer

    def forward(self, code_input_ids, nl_input_ids, labels, return_vec=False):
        batch_size = code_input_ids.size(0)
        input_ids = torch.cat([code_input_ids, nl_input_ids], dim=0)
        # build input dict
        input_dict = prepare_input_dict_for_representation(input_ids,
                                                           model_type=self.args.model_type,
                                                           pad_token_id=self.tokenizer.pad_token_id)
        # get the model-specific representation vector
        vectors = get_representation_vector(self.args,
                                            model=self.model,
                                            input_dict=input_dict,
                                            eos_token_id=self.tokenizer.eos_token_id)

        code_vec = vectors[:batch_size]
        nl_vec = vectors[batch_size:]
        if return_vec:
            return code_vec, nl_vec

        nl_vec = nl_vec.unsqueeze(1).repeat([1, batch_size, 1])
        code_vec = code_vec.unsqueeze(0).repeat([batch_size, 1, 1])
        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec - code_vec, nl_vec * code_vec), 2)).squeeze(2)    # [B, B]
        matrix_labels = torch.diag(labels).float()  # (Batch, Batch)
        poss = logits[matrix_labels == 1]
        negs = logits[matrix_labels == 0]

        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        loss = - (torch.log(1 - negative_pairs).mean() + torch.log(positive_pairs).mean())
        predictions = (logits.gather(0, torch.arange(batch_size, device=loss.device).unsqueeze(0)).squeeze(0) > 0.5)\
            .int()
        return {"loss": loss, "predictions": predictions}


def count_params(model):
    """Count the number of learnable parameters of given model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def layer_wise_parameters(model):
    """Returns a printable table representing the layer-wise model parameters, their shapes and numbers"""
    table = PrettyTable()
    table.field_names = ["Layer Name", "Output Shape", "Param #"]
    table.align["Layer Name"] = "l"
    table.align["Output Shape"] = "r"
    table.align["Param #"] = "r"
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            table.add_row([name, str(list(parameters.shape)), parameters.numel()])
    return table


wrap_model_to_class = {
    "retrieval": RetrievalModel,
    "search": SearchModel,
    "cosqa": CoSQAModel
}


def build_model_tokenizer(args) -> (PreTrainedModel, PreTrainedTokenizer):
    """Builds the model and tokenizer."""

    # load config
    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    logger.info(f"Loaded config '{config.__class__.__name__}' from '{args.model_name}'")
    logger.debug(config)

    # load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.model_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    args.pad_token_id = tokenizer.pad_token_id

    logger.info(f"Loaded tokenizer '{tokenizer.__class__.__name__}' from '{args.tokenizer_name}', "
                f"size: {len(tokenizer)}")
    logger.debug(f"Special symbols: {tokenizer.all_special_tokens}")

    # load unwrapped model
    model_class = configs.MODEL_TYPE_TO_CLASS[args.task_type][args.model_type]
    if args.train_from_scratch:
        model = model_class(config)
    else:
        model = model_class.from_pretrained(args.model_name, config=config)
        if args.task in configs.TASK_TYPE_TO_LIST["classification"]:
            if args.model_type == "bert":
                model.classifier = nn.Linear(config.hidden_size, config.num_labels)
            elif args.model_type == "roberta":
                model.classifier = RobertaClassificationHead(config)
            elif args.model_type == "gpt2":
                model.score = nn.Linear(config.n_embd, config.num_labels, bias=False)
            elif args.model_type in ["bart", "plbart"]:
                model.classification_head = BartClassificationHead(
                    config.d_model,
                    config.d_model,
                    config.num_labels,
                    config.classifier_dropout
                )
                model.model._init_weights(model.classification_head.dense)
                model.model._init_weights(model.classification_head.out_proj)

    logger.info(f"Loaded unwrapped model '{model.__class__.__name__}' from '{args.model_name}'")

    # wrap model
    # build seq2seq model for bert/roberta
    if args.task in configs.TASK_TYPE_TO_LIST["seq2seq"] and \
            args.model_type in ["bert", "roberta"]:
        logger.info(f"Trainable parameters: {human_format(count_params(model))}")
        seq2seq_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
        model = EncoderDecoderModel(config, encoder=model)
        config_decoder = seq2seq_config.decoder
        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        logger.info(f"Built seq2seq model for '{model.__class__.__name__}'")
    elif args.task in wrap_model_to_class.keys():
        logger.info(f"Trainable parameters: {human_format(count_params(model))}")
        model = wrap_model_to_class[args.task](args, model=model, tokenizer=tokenizer)

    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Loaded model '{model.__class__.__name__}' from '{args.model_name}'")
    logger.info(f"Trainable parameters: {human_format(count_params(model))}")
    logger.debug(f"Layer-wise parameters:\n{layer_wise_parameters(model)}")

    return model, tokenizer


def prepare_model_kwargs(args, batch):

    model_kwargs = {}

    if args.model_type in ["t5", "codet5"] and args.task in configs.TASK_TYPE_TO_LIST["classification"]:
        model_kwargs["input_ids"] = batch[0]
        model_kwargs["attention_mask"] = batch[0].ne(args.pad_token_id)
        model_kwargs["labels"] = batch[1]
        model_kwargs["decoder_attention_mask"] = batch[1].ne(args.pad_token_id)

    elif args.task in configs.TASK_TYPE_TO_LIST["classification"]:
        model_kwargs["input_ids"] = batch[0]
        model_kwargs["attention_mask"] = batch[0].ne(args.pad_token_id)
        model_kwargs["labels"] = batch[1]

    elif args.task == "retrieval":
        model_kwargs["input_ids"] = batch[0]
        model_kwargs["pos_input_ids"] = batch[1]
        model_kwargs["neg_input_ids"] = batch[2]
        model_kwargs["labels"] = batch[3]

    elif args.task == "search":
        model_kwargs["code_input_ids"] = batch[0]
        model_kwargs["nl_input_ids"] = batch[1]

    elif args.task == "cosqa":
        model_kwargs["code_input_ids"] = batch[0]
        model_kwargs["nl_input_ids"] = batch[1]
        model_kwargs["labels"] = batch[2]

    elif args.task in configs.TASK_TYPE_TO_LIST["seq2seq"]:
        model_kwargs["input_ids"] = batch[0]
        model_kwargs["attention_mask"] = batch[0].ne(args.pad_token_id)
        model_kwargs["labels"] = batch[1]
        model_kwargs["decoder_attention_mask"] = batch[1].ne(args.pad_token_id)

    elif args.task == "completion":
        pass

    return model_kwargs

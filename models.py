from transformers import AutoTokenizer, AutoConfig, EncoderDecoderConfig, EncoderDecoderModel
import logging

import configs
from utils import human_format


logger = logging.getLogger(__name__)


def count_params(model):
    """Count the number of learnable parameters of given model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model_tokenizer(args):

    config = AutoConfig.from_pretrained(args.model_name)
    logger.info(f"Loaded config '{config.__class__}' from '{args.model_name}'")
    logger.debug(config)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if args.model_type == 'gpt2':
        tokenizer.pad_token = tokenizer.eos_token
    logger.info(f"Loaded tokenizer '{tokenizer.__class__}' from '{args.tokenizer_name}', size: {len(tokenizer)}")
    logger.debug(f"Special symbols: {tokenizer.all_special_tokens}")

    # load model
    model_class = configs.MODEL_TYPE_TO_CLASS[args.task_type][args.model_type]
    if args.train_from_scratch:
        model = model_class(config)
    else:
        model = model_class.from_pretrained(args.model_name)
    logger.info(f"Loaded model '{model.__class__}' from '{args.model_name}'")

    # build seq2seq model for bert/roberta
    if args.task_type == "seq2seq" and args.model_type in ["bert", "roberta"]:
        logger.info(f"Trainable parameters: {human_format(count_params(model))}")
        seq2seq_config = EncoderDecoderConfig.from_encoder_decoder_configs(config, config)
        model = EncoderDecoderModel(config, encoder=model)
        config_decoder = seq2seq_config.decoder
        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        model.config.decoder_start_token_id = tokenizer.cls_token_id
        logger.info(f"Built seq2seq model for '{model.__class__}'")

    model.resize_token_embeddings(len(tokenizer))
    logger.info(f"Loaded model '{model.__class__}' from '{args.model_name}'")
    logger.info(f"Trainable parameters: {human_format(count_params(model))}")

    return model, tokenizer

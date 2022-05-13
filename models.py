from transformers import AutoTokenizer, \
    BertConfig, BertForSequenceClassification, BertTokenizer, \
    RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, \
    GPT2Config, GPT2ForSequenceClassification, GPT2Tokenizer, \
    BartConfig, BartForSequenceClassification, BartTokenizer, \
    T5Config, T5ForConditionalGeneration, T5Tokenizer, \
    PLBartConfig, PLBartModel, PLBartTokenizer

import configs


def init_model_tokenizer(args):

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    if args.model_type == "bert":
        pass

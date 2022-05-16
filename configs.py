
from transformers import \
    BertForSequenceClassification, BertModel, \
    RobertaForSequenceClassification, RobertaModel, \
    GPT2ForSequenceClassification, GPT2LMHeadModel, \
    BartForSequenceClassification, BartForConditionalGeneration, \
    T5ForConditionalGeneration, \
    PLBartForSequenceClassification, PLBartForConditionalGeneration


# all tasks
# TASK_NAME_TO_TYPE = {
#     # classification task
#     "defect": "classification",
#     "clone": "classification",
#     "exception": "classification",
#     # retrieval task
#     "retrieval": "retrieval",
#     "search": "retrieval",
#     # CoCLR training approach for CoSQA
#     "cosqa": "coclr",
#     # sequence-to-sequence
#     "translation": "seq2seq",
#     "fixing": "seq2seq",
#     "mutant": "seq2seq",
#     "assert": "seq2seq",
#     "summarization": "seq2seq",
#     "generation": "seq2seq",
#     "completion": "casual"
# }

# map the task to datasets
TASK_TO_DATASET = {
    "defect": ["devign"],
    "clone": ["bigclonebench"],
    "exception": ["exception"],
    "retrieval": ["poj104"],
    "search": ["advtest"],
    "cosqa": ["cosqa"],
    "translation": ["codetrans"],
    "fixing": ["bfp"],
    "completion": [],
    "mutant": ["mutant"],
    "assert": ["assert"],
    "summarization": ["codesearchnet"],
    "generation": ["concode"]
}

# map the dataset to sub-tasks
DATASET_TO_SUBSET = {
    "codetrans": ["java-cs", "cs-java"],
    "bfp": ["small", "medium"],
    "assert": ["abs", "raw"],
    "codesearchnet": ["java", "python", "javascript", "php", "go", "ruby"]
}

# map the task to its major metric
TASK_TO_MAJOR_METRIC = {
    "defect": "acc",
    "clone": "f1",
    "exception": "acc",
    "retrieval": "map",
    "search": "mrr",
    "cosqa": "acc",
    "translation": "em",
    "fixing": "em",
    "completion": "",
    "mutant": "em",
    "assert": "em",
    "summarization": "bleu",
    "generation": "em"
}

# map the model identifier to (base model type, model path, tokenizer path)
MODEL_ID_TO_NAMES = {
    # vanilla transformer
    "transformer": ("roberta", "none", "microsoft/codebert-base"),
    # nlp pre-trained models
    "bert": ("bert", "bert-base-uncased", "bert-base-uncased"),
    "roberta": ("roberta", "roberta-base", "roberta-base"),
    "gpt2": ("gpt2", "distilgpt2", "distillgpt2"),
    "bart": ("bart", "facebook/bart-base", "facebook/bart-base"),
    "t5": ("t5", "t5-base", "t5-base"),
    # pre-trained models of source code
    "codebert": ("robert", "microsoft/codebert-base", "microsoft/codebert-base"),
    "graphcodebert": ("roberta", "microsoft/graphcodebert-base", "microsoft/graphcodebert-base"),
    "javabert": ("bert", "CAUKiel/JavaBERT", "CAUKiel/JavaBERT"),
    "codegpt": ("gpt2", "microsoft/CodeGPT-small-java", "microsoft/CodeGPT-small-java"),
    "codegpt-adapted": ("gpt2", "microsoft/CodeGPT-small-java-adaptedGPT2", "microsoft/CodeGPT-small-java-adaptedGPT2"),
    "plbart": ("plbart", "uclanlp/plbart-base", "uclanlp/plbart-base"),
    "cotext": ("t5", "razent/cotext-2-cc", "razent/cotext-2-cc"),
    "codet5": ("codet5", "Salesforce/codet5-base", "Salesforce/codet5-base"),
    "unixcoder": ("roberta", "microsoft/unixcoder-base", "microsoft/unixcoder-base")
}

# task type to model class
MODEL_TYPE_TO_CLASS = {
    "classification": {
        "bert": BertForSequenceClassification,
        "roberta": RobertaForSequenceClassification,
        "gpt2": GPT2ForSequenceClassification,
        "bart": BartForSequenceClassification,
        "t5": T5ForConditionalGeneration,
        "plbart": PLBartForSequenceClassification,
        "codet5": T5ForConditionalGeneration
    },
    "seq2seq": {
        "bert": BertModel,
        "roberta": RobertaModel,
        "gpt2": GPT2LMHeadModel,
        "bart": BartForConditionalGeneration,
        "t5": T5ForConditionalGeneration,
        "plbart": PLBartForConditionalGeneration,
        "codet5": T5ForConditionalGeneration
    },
    "casual": {

    }
}

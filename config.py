

# map the model identifier to (base model type, model path, tokenizer path)
MODEL_ID_TO_CLASS = {
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

# all tasks
TASK_NAME_TO_TYPE = {
    # classification task
    "defect": "classification",
    "clone": "classification",
    "exception": "classification",
    # retrieval task
    "retrieval": "retrieval",
    "search": "retrieval",
    # CoCLR training approach for CoSQA
    "cosqa": "coclr",
    # sequence-to-sequence
    "translation": "seq2seq",
    "bug": "seq2seq",
    "mutant": "seq2seq",
    "assert": "seq2seq",
    "summarization": "seq2seq",
    "generation": "seq2seq",
    "completion": "generative"
}

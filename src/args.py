from transformers import SchedulerType
from argparse import ArgumentParser

import configs


def add_args(parser: ArgumentParser):

    # model identifier
    parser.add_argument("--model", type=str, default="roberta",
                        choices=configs.MODEL_ID_TO_NAMES.keys(),
                        help="Model identifier.")

    # task, dataset and subtask
    parser.add_argument("--task", type=str, default="defect",
                        choices=configs.TASK_TO_DATASET.keys(),
                        help="Task name.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name, leave empty for default.")
    parser.add_argument("--subset", type=str, default=None,
                        help="The subset name, if any.")
    parser.add_argument("--data-dir", type=str, default="../datasets",
                        help="The directory to store datasets.")

    # train, valid and test procedure
    parser.add_argument("--only-test", action="store_true", default=False,
                        help="Whether to only perform testing procedure.")
    parser.add_argument("--do-not-valid", action="store_true", default=False,
                        help="Do not do validation after each epoch.")

    # hyper parameters
    parser.add_argument("--override-params", action="store_true", default=False,
                        help="Override pre-defined task-specific hyperparameter settings.")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Number of total training epochs.")
    parser.add_argument("--train-batch-size", type=int, default=None,
                        help="Size of training batch, per device.")
    parser.add_argument("--eval-batch-size", type=int, default=None,
                        help="Size of validation/testing batch, per device.")
    parser.add_argument("--max-source-length", type=int, default=None,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max-source-pair-length", type=int, default=None,
                        help="The maximum total source pair sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max-target-length", type=int, default=None,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning-rate", type=float, default=5e-5,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam-size", type=int, default=5,
                        help="beam size for beam search.")
    parser.add_argument("--weight-decay", type=float, default=0.0,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam-epsilon", type=float, default=1e-8,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Max gradient norm, 0 to disable.")
    parser.add_argument("--num-warmup-steps", type=int, default=None,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed, -1 to disable.")
    parser.add_argument("--lr-scheduler-type", type=SchedulerType, default="linear",
                        help="The scheduler type to use.",
                        choices=["linear", "cosine", "cosine_with_restarts", "polynomial",
                                 "constant", "constant_with_warmup"])

    # environment
    parser.add_argument("--cuda-visible-devices", type=str, default=None,
                        help='Index (Indices) of the GPU to use in a cluster.')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable cuda, overrides cuda-visible-devices.")
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision option, chosen from `no`, `fp16`, `bf16`")

    # limitations
    parser.add_argument("--max-train-steps", type=int, default=None,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--training-sample", type=float, default=None,
                        help="Whether to sample a specific ratio (when between 0 and 1) or number (when >=0) "
                             "of training instance for training.")
    parser.add_argument("--train-from-scratch", action="store_true", default=False,
                        help="Whether to fine-tune from scratch, will not load pre-trained models.")

    # outputs and savings
    parser.add_argument("--run-name", type=str, default=None,
                        help="Unique name of current running, will be automatically set if it is None.")
    parser.add_argument("--wandb-mode", type=str, default="disabled",
                        choices=["online", "offline", "disabled"],
                        help="Set the wandb mode.")


def set_task_hyper_parameters(args):

    num_epochs = 30
    train_batch_size = 32
    eval_batch_size = 32
    max_source_length = None
    max_source_pair_length = None
    max_target_length = None
    learning_rate = 5e-5
    num_warmup_steps = 100
    patience = 5

    if args.task == "defect":
        num_epochs = 20
        max_source_length = 512
        max_target_length = 3
        patience = 5

    elif args.task == "clone":
        num_epochs = 1
        max_source_length = 256
        max_source_pair_length = 256
        max_target_length = 3
        patience = 2

    elif args.task == "exception":
        num_epochs = 20
        max_source_length = 256
        max_target_length = 10
        patience = 5

    elif args.task == "retrieval":
        num_epochs = 20
        max_source_length = 512
        max_target_length = 512
        patience = 5

    elif args.task == "search":
        num_epochs = 20
        max_source_length = 256
        max_target_length = 256
        patience = 5

    elif args.task == "cosqa":
        num_epochs = 5
        learning_rate = 5e-6
        num_warmup_steps = 500
        max_source_length = 256

    elif args.task == "translation":
        num_epochs = 50
        max_source_length = 384
        max_target_length = 256
        patience = 5

    elif args.task == "fixing":
        num_epochs = 50
        if args.sub_task == "small":
            max_source_length = 128
            max_target_length = 128
        elif args.sub_task == "medium":
            max_source_length = 256
            max_target_length = 256
        patience = 5

    elif args.task == "mutant":
        num_epochs = 50
        max_source_length = 128
        max_target_length = 128
        patience = 5

    elif args.task == "assert":
        num_epochs = 30
        if args.sub_task == "abs":
            max_source_length = 512
            max_target_length = 64
        elif args.sub_task == "raw":
            max_source_length = 256
            max_target_length = 32
        patience = 5

    elif args.task == "summarization":
        num_epochs = 15
        max_source_length = 256
        max_target_length = 128
        patience = 3

    elif args.task == "generation":
        num_epochs = 30
        max_source_length = 256
        max_target_length = 128
        patience = 5

    args.max_source_length = max_source_length
    args.max_source_pair_length = max_source_pair_length
    args.max_target_length = max_target_length
    if not args.override_params:
        args.num_epochs = num_epochs
        args.train_batch_size = train_batch_size
        args.eval_batch_size = eval_batch_size
        args.learning_rate = learning_rate
        args.num_warmup_steps = num_warmup_steps
        args.patience = patience


def check_args(args):
    """Check if args values are valid, and conduct some default settings."""

    # task major metric
    args.major_metric = configs.TASK_TO_MAJOR_METRIC[args.task]
    # task type
    args.task_type = configs.TASK_NAME_TO_TYPE[args.task]

    # dataset
    dataset_list = configs.TASK_TO_DATASET[args.task]
    assert len(dataset_list) != 0, f'There is no dataset configured as the dataset of `{args.task}`.'
    if args.dataset is None:
        if len(dataset_list) > 1:
            raise ValueError(f"Please specific a dataset of task `{args.task}` "
                             f"when more than one datasets is configured.")
        else:
            args.dataset = dataset_list[0]
    else:
        assert args.dataset in dataset_list, \
            f'Dataset `{args.dataset}` is not configured as the dataset of task `{args.task}`.'

    # subset
    if args.subset is None:
        assert args.dataset not in configs.DATASET_TO_SUBSET, \
            f"Please specific a subset of dataset `{args.dataset}` when it has multiple subsets."
    else:
        assert args.dataset in configs.DATASET_TO_SUBSET, \
            f"Dataset `{args.dataset}` has no subset."
        assert args.subtask in configs.DATASET_TO_SUBSET[args.dataset], \
            f"Dataset `{args.dataset}` has not subset called `{args.subset}`"

    # set language
    args.source_lang = None
    args.target_lang = None
    if args.dataset == "devign":
        args.source_lang = "c"
        args.target_lang = None
    elif args.dataset == "bigclonebench":
        args.source_lang = "java"
    elif args.dataset == "exception":
        args.source_lang = "python"
    elif args.dataset == "poj104":
        args.source_lang = "c"
        args.target_lang = "c"
    elif args.dataset == "advtest":
        args.source_lang = "en"
        args.target_lang = "python"
    elif args.dataset == "cosqa":
        args.source_lang = "python"
        args.target_lang = "en"
    elif args.dataset == "codetrans":
        args.source_lang, args.target_lang = args.subtask.split("-")
    elif args.dataset == "bfp":
        args.source_lang = "java"
        args.target_lang = "java"
    elif args.dataset == "mutant":
        args.source_lang = "java"
        args.target_lang = "java"
    elif args.dataset == "assert":
        args.source_lang = "java"
        args.target_lang = "java"
    elif args.dataset == "codesearchnet":
        args.source_lang = args.subtask
        args.target_lang = "en"
    elif args.dataset == "concode":
        args.source_lang = "en"
        args.target_lang = "c"

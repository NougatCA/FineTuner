
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
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name, leave empty for default.")
    parser.add_argument("--subset", type=str, default="",
                        help="The subset name, if any.")
    parser.add_argument("--data-dir", type=str, default="./datasets",
                        help="The directory to store datasets.")

    # train, valid and test procedure
    parser.add_argument("--do-train", action="store_true",
                        help="Whether to perform training procedure.")
    parser.add_argument("--do-valid", action="store_true",
                        help="Whether to perform validation procedure during training.")
    parser.add_argument("--do-test", action="store_true",
                        help="Whether to perform testing procedure.")

    # hyper parameters
    parser.add_argument("--override-params", action="store_true", default=False,
                        help="Override pre-defined task-specific hyperparameter settings.")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Number of total training epochs.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Size of mini-batch, per device.")
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
                        help="Max gradient norm.")
    parser.add_argument("--warmup-steps", type=int, default=None,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--patience", type=int, default=None,
                        help="Early stopping patience.")
    parser.add_argument("--random-seed", type=int, default=42,
                        help="Random seed, -1 to disable.")

    # environment
    parser.add_argument("--cuda-visible-devices", type=str, default=None,
                        help='Index (Indices) of the GPU to use in a cluster.')
    parser.add_argument("--no-cuda", action="store_true",
                        help="Disable cuda, overrides cuda-visible-devices.")
    parser.add_argument("--mixed-precision", type=str, default="fp16",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision option, chosen from `no`, `fp16`, `bf16`")

    # limitations
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--max-data-num", type=int, default=-1,
                        help='Max number of data instances to use, -1 for full data.')
    parser.add_argument("--training-sample", type=float, default=0,
                        help="Whether to sample a specific ratio (when between 0 and 1) or number (when >=0) "
                             "of training instance for training.")
    parser.add_argument("--train-from-scratch", action="store_true", default=False,
                        help="Whether to fine-tune from scratch, will not load pre-trained models.")

    # outputs and savings
    parser.add_argument("--run-name", type=str, default=None,
                        help="Unique name of current running, will be automatically set if it is None.")
    parser.add_argument("--wandb-offline", action="store_true", default=False,
                        help="Set the wandb mode to offline so that the logging will not be uploaded to the server.")


def set_task_hyper_parameters(args):

    num_epochs = 30
    batch_size = 64
    max_source_length = None
    max_source_pair_length = None
    max_target_length = None
    learning_rate = 5e-5
    warmup_steps = 100
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
        warmup_steps = 500
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
        args.batch_size = batch_size
        args.learning_rate = learning_rate
        args.warmup_steps = warmup_steps
        args.patience = patience


def check_args(args):
    """Check if args values are valid, and conduct some default settings."""

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
    if args.subtask is None:
        assert args.dataset not in configs.DATASET_TO_SUBSET, \
            f"Please specific a subset of dataset `{args.dataset}` when it has multiple subsets."
    else:
        assert args.dataset in configs.DATASET_TO_SUBSET, \
            f"Dataset `{args.dataset}` has no subset."
        assert args.subtask in configs.DATASET_TO_SUBSET[args.dataset], \
            f"Dataset `{args.dataset}` has not subset called `{args.subset}`"

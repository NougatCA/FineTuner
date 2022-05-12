
from argparse import ArgumentParser

import config


def add_args(parser: ArgumentParser):

    # model identifier
    parser.add_argument("--model", type=str, default="roberta",
                        choices=config.MODEL_ID_TO_CLASS.keys(),
                        help="Model identifier.")

    # task, dataset and subtask
    parser.add_argument("--task", type=str, default="defect",
                        choices=config.TASK_NAME_TO_TYPE.keys(),
                        help="Task name.")
    parser.add_argument("--dataset", type=str, default="",
                        help="Dataset name, leave empty for default.")
    parser.add_argument("--sub-task", type=str, default="",
                        help="The subtask name, if any.")

    # train, valid and test procedure
    parser.add_argument("--do-train", action="store_true",
                        help="Whether to perform training procedure.")
    parser.add_argument("--do-valid", action="store_true",
                        help="Whether to perform validation procedure during training.")
    parser.add_argument("--do-test", action="store_true",
                        help="Whether to perform testing procedure.")

    # hyper parameters
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Number of total training epochs.")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Size of mini-batch, per device.")
    parser.add_argument("--max-source-length", type=int, default=None,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
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
    parser.add_argument("--warmup-steps", type=int, default=100,
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

    # limitations
    parser.add_argument("--max_steps", type=int, default=-1,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--max-data-num", type=int, default=-1,
                        help='Max number of data instances to use, -1 for full data.')

    # outputs and savings
    parser.add_argument("--run-name", type=str, default=None,
                        help="Unique name of current running, will be automatically set if it is None.")


def check_args(args):
    pass

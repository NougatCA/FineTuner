
from argparse import ArgumentParser

import config


def add_args(parser: ArgumentParser):

    parser.add_argument("--model",
                        type=str,
                        default="roberta",
                        choices=config.MODEL_ID_TO_CLASS.keys(),
                        help="Model identifier")

    parser.add_argument("--task",
                        type=str,
                        default="defect",
                        choices=config.TASK_NAME_TO_TYPE.keys(),
                        help="Task name")

    parser.add_argument("--dataset", type=str, default="", help="Dataset name")
    parser.add_argument("--subtask", type=str, default="", help="The sub-task name, if any")

    parser.add_argument("--do-train", action="store_true", help="Whether to perform training procedure")
    parser.add_argument("--do-valid", action="store_true", help="Whether to perform validation procedure")
    parser.add_argument("--do-test", action="store_true", help="Whether to perform testing procedure")

    parser.add_argument("--num-epoch", type=int, default=10, help="Number of total training epochs")


import argparse

from args import add_args, check_args
from utils import get_run_name


def main():
    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)

    args = parser.parse_args()

    # check args
    check_args(args)

    # prepare some preliminary arguments
    if args.run_name is None:
        args.run_name = get_run_name(args)


if __name__ == "__main__":
    main()

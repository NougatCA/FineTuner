
import argparse

from args import add_args


def main():
    # parse arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.register('type', 'bool', lambda v: v.lower() in ['yes', 'true', 't', '1', 'y'])

    add_args(parser)

    args = parser.parse_args()

    print(args.do_train)


if __name__ == "__main__":
    main()

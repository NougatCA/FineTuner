
from models import init_model_tokenizer
from data import load_examples


def run_fine_tune(args):

    model, tokenizer = init_model_tokenizer(args)

    if args.do_train:
        train_examples = load_examples()

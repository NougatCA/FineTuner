
from models import init_model_tokenizer
from data import load_examples, load_aux_data


def run_fine_tune(args):

    model, tokenizer = init_model_tokenizer(args)

    aux_data = None
    if args.dataset in ["bigclonebench"]:
        aux_data = load_aux_data()

    if args.do_train:
        train_examples = load_examples()

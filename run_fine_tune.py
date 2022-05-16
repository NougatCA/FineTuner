
from models import init_model_tokenizer
from data import load_examples, load_aux_data, create_dataset


def run_fine_tune(args):

    model, tokenizer = init_model_tokenizer(args)

    aux_data = None
    if args.task in ["exception"] or args.dataset in ["bigclonebench"]:
        aux_data = load_aux_data(args)

    # prepare data for training and validation
    if args.do_train:
        train_examples = load_examples(args, split="train", aux_data=aux_data)
        train_dataset = create_dataset(args, examples=train_examples, tokenizer=tokenizer, split="train")
    if args.do_valid:
        valid_examples = load_examples(args, split="valid", aux_data=aux_data)

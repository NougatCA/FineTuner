
def get_run_name(args):
    tokens = [args.model, args.task, args.dataset, args.sub_task,
              f"bs{args.batch_size}", f"ep{args.num_epochs}", f"lr{args.learning_rate}", f"warmup{args.warmup_steps}"]
    return "_".join([token for token in tokens if token is not None or token != ""])


def get_short_run_name(args):
    tokens = [args.model, args.task, args.dataset, args.sub_task]
    return "_".join([token for token in tokens if token is not None or token != ""])


def human_format(num):
    """Transfer count number."""
    num = float('{:.3g}'.format(num))
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    return '{}{}'.format('{:f}'.format(num).rstrip('0').rstrip('.'),
                         ['', 'K', 'M', 'B', 'T'][magnitude])


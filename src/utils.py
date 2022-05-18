
from prettytable import PrettyTable


class EarlyStopController(object):
    """
    A controller for early stopping.
    Args:
        patience (int):
            Maximum number of consecutive epochs without breaking the best record.
        higher_is_better (bool, optional, defaults to True):
            Whether a higher record is seen as a better one.
    """

    def __init__(self, patience: int, higher_is_better=True):
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.early_stop = False
        self.hit = False
        self.counter = 0

        self.best = None
        self.best_model = None
        self.best_epoch = None

    def __call__(self, score: float, model, epoch: int):
        """Calls this after getting the validation metric each epoch."""
        # first calls
        if self.best is None:
            self.best = score
            self.model = model
            self.hit = True
            self.best_epoch = epoch
        else:
            # not hits the best record
            if (self.higher_is_better and score < self.best) or (not self.higher_is_better and score > self.best):
                self.hit = False
                self.counter += 1
                if self.counter > self.patience:
                    self.early_stop = True
            # hits the best record
            else:
                self.hit = True
                self.counter = 0
                self.best = score
                self.best_model = model
                self.best_epoch = epoch


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


def postprocess_results(result_dict: dict, major_metric=None):
    """
    Post-processes the evaluation result dict, such as generates result table and extracts major score.

    Args:
        result_dict (dict):
            A dict mapping from metric name to its score.
        major_metric (str, optional, defaults to None):
            The major metric name, if given, will return the major score.
    Returns:
        result_table (PrettyTable):
            A table of results.
        major_score (Union[None, float])
            The score corresponds to the major metric, `None` if `major_metric` is `None`.
    """
    results_table = PrettyTable()
    results_table.field_names = ["Metric", "Score"]
    results_table.align["Metric"] = "c"
    results_table.align["Score"] = "l"
    major_score = None
    for metric, score in result_dict.items():
        if major_metric and metric.endswith(major_metric):
            results_table.add_row([f"**{metric}**", score])
            major_score = score
        else:
            results_table.add_row([metric, str(score)])
    return results_table, major_score

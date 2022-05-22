import torch
import torch.nn as nn
from prettytable import PrettyTable
from dataclasses import dataclass


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


@dataclass
class LabelSmoother:
    """
    Adds label-smoothing on a pre-computed output from a Transformers model.

    Args:
        epsilon (`float`, *optional*, defaults to 0.1):
            The label smoothing factor.
        ignore_index (`int`, *optional*, defaults to -100):
            The index in the labels to ignore when computing the loss.
    """

    epsilon: float = 0.1
    ignore_index: int = -100

    def __call__(self, model_output, labels):
        logits = model_output["logits"] if isinstance(model_output, dict) else model_output[0]
        log_probs = -nn.functional.log_softmax(logits, dim=-1)
        if labels.dim() == log_probs.dim() - 1:
            labels = labels.unsqueeze(-1)

        padding_mask = labels.eq(self.ignore_index)
        # In case the ignore_index is -100, the gather will fail, so we replace labels by 0. The padding_mask
        # will ignore them in any case.
        labels = torch.clamp(labels, min=0)
        nll_loss = log_probs.gather(dim=-1, index=labels)
        # works for fp16 input tensor too, by internally upcasting it to fp32
        smoothed_loss = log_probs.sum(dim=-1, keepdim=True, dtype=torch.float32)

        nll_loss.masked_fill_(padding_mask, 0.0)
        smoothed_loss.masked_fill_(padding_mask, 0.0)

        # Take the mean over the label dimensions, then divide by the number of active elements (i.e. not-padded):
        num_active_elements = padding_mask.numel() - padding_mask.long().sum()
        nll_loss = nll_loss.sum() / num_active_elements
        smoothed_loss = smoothed_loss.sum() / (num_active_elements * log_probs.shape[-1])
        return (1 - self.epsilon) * nll_loss + self.epsilon * smoothed_loss


def get_run_name(args):
    tokens = [args.model, args.task, args.dataset, args.subset,
              f"bs{args.train_batch_size}", f"ep{args.num_epochs}",
              f"lr{args.learning_rate}", f"warmup{args.num_warmup_steps}"]
    return "_".join([token for token in tokens if token is not None and token != ""])


def get_short_run_name(args):
    tokens = [args.model, args.task, args.dataset, args.subset]
    return "_".join([token for token in tokens if token is not None and token != ""])


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


def layer_wise_parameters(model):
    """Returns a printable table representing the layer-wise model parameters, their shapes and numbers"""
    table = PrettyTable()
    table.field_names = ["Layer Name", "Output Shape", "Param #"]
    table.align["Layer Name"] = "l"
    table.align["Output Shape"] = "r"
    table.align["Param #"] = "r"
    for name, parameters in model.named_parameters():
        if parameters.requires_grad:
            table.add_row([name, str(list(parameters.shape)), parameters.numel()])
    return table

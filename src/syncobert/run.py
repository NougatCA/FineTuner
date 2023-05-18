import logging
import sys
import warnings

import accelerate
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler
from transformers import get_linear_schedule_with_warmup

from syncobert.data import (PretrainDataset, Scene1Dataset, Scene2Dataset,
                            Scene3Dataset)
from syncobert.model import TaskModel, init_model

warnings.filterwarnings('ignore')


class IgnoreFilter(logging.Filter):
    def filter(self, record):
        return 'syncobert' not in record.getMessage()


def get_logger():
    accelerate_logger = accelerate.logging.get_logger(
        'syncobert', log_level="INFO")
    accelerate_logger.logger.addFilter(IgnoreFilter())

    ch = logging.StreamHandler()
    fh = logging.FileHandler('syncobert.log')
    accelerate_logger.logger.addHandler(ch)
    accelerate_logger.logger.addHandler(fh)

    file_formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")
    console_formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(lineno)s %(message)s")

    ch.setFormatter(console_formatter)
    fh.setFormatter(file_formatter)

    return accelerate_logger


logger = get_logger()


def train(model, accelerator, dataset):
    scene1_datset = Scene1Dataset(dataset)
    scene2_datset = Scene2Dataset(dataset)
    scene3_datset = Scene3Dataset(dataset)
    scene1_dataloader = DataLoader(scene1_datset, sampler=RandomSampler(
        scene1_datset), batch_size=16, num_workers=16)
    scene1_dataloader_iter = iter(scene1_dataloader)
    scene2_dataloader = DataLoader(scene2_datset, sampler=RandomSampler(
        scene2_datset), batch_size=16, num_workers=16)
    scene2_dataloader_iter = iter(scene2_dataloader)
    scene3_dataloader = DataLoader(scene3_datset, sampler=RandomSampler(
        scene3_datset), batch_size=16, num_workers=16)
    scene3_dataloader_iter = iter(scene3_dataloader)

    optimizer = AdamW(model.parameters(), lr=1e-4, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=330000)

    optimizer, scene1_dataloader, scene2_dataloader, scene3_dataloader, scheduler = accelerator.prepare(optimizer,
                                                                                                        scene1_dataloader,
                                                                                                        scene2_dataloader,
                                                                                                        scene3_dataloader,
                                                                                                        scheduler)
    logger.info('start training')
    model.train()

    for step in range(330000):
        if step % 3 == 0:
            batch = next(scene1_dataloader_iter)
            input_io = {
                'masked_ids': batch[0],
                'labels': batch[1],
                'code_pos': batch[2],
                'ast_pos': batch[3],
                'edge_sim_mat': batch[4],
                'is_identi': batch[5],
                'attn_mask': batch[6],
            }
            loss = model(input_io, scene='first')
        elif step % 3 == 1:
            batch = next(scene2_dataloader_iter)
            input_io = {
                'masked_nl_ids': batch[0],
                'masked_code_ids': batch[1],
            }
            loss = model(input_io, scene='second')
        elif step % 3 == 2:
            batch = next(scene3_dataloader_iter)
            input_io = {
                'first_ids': batch[0],
                'second_ids': batch[1],
            }
            loss = model(input_io, scene='third')

        accelerator.backward(loss / 16.)
        if step % 16 == 0:
            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            logger.info(f'step: {step}, loss: {loss.item()}')


if __name__ == '__main__':
    accelerater = Accelerator(split_batches=True,
                              kwargs_handlers=[
                                  DistributedDataParallelKwargs(bucket_cap_mb=15, find_unused_parameters=True)])

    base_model, tokenizer = init_model()
    dataset = PretrainDataset(tokenizer).load('/datasets/pre_train.pkl')
    unwrap_model = TaskModel(base_model, 0.1, tokenizer.vocab_size)
    model = accelerater.prepare(unwrap_model)

    train(model, accelerater, dataset)
    unwrap_model.syncobert.save_model('/syncobert/syncobert.bin')

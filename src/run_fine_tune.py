import json
import os
import torch
from torch.optim import AdamW
from transformers import get_scheduler
import logging
import math
from tqdm import tqdm
import numpy as np

from models import build_model_tokenizer, prepare_model_kwargs
from data import prepare_data
import configs
from eval import acc_and_f1, map_score, mrr
from utils import EarlyStopController, postprocess_results

logger = logging.getLogger(__name__)


def run_eval(args, model, tokenizer, dataloader, split, raw_examples, epoch=None) -> dict:
    assert split in ["valid", "test"]
    assert epoch or split == "test"
    eval_bar = tqdm(dataloader, total=len(dataloader), desc="Validating" if split == "valid" else "Testing")

    # general statistics
    num_examples = 0
    num_steps = 0
    loss_list = []

    results = {}
    save_dir = os.path.join(args.eval_dir, f"valid_epoch_{epoch}" if split == "valid" else "test")
    os.makedirs(save_dir)

    model.eval()
    if args.task in configs.TASK_TYPE_TO_LIST["classification"] or args.task == "cosqa":
        preds = []
        golds = []
        for batch in eval_bar:
            with torch.no_grad():
                if args.task == "cosqa":
                    code_input_ids, nl_input_ids, labels = batch
                    outputs = model(code_input_ids, nl_input_ids, labels)
                else:
                    input_ids, labels = batch
                    outputs = model(input_ids, labels=labels)

                loss = outputs.loss.mean().item()
                loss_list.append(loss)

                if args.task == "cosqa":
                    preds.extend(outputs.predictions.cpu().tolist())
                else:
                    preds.extend(np.argmax(outputs.logits.cpu().tolist(), axis=1))
                golds.extend(labels.cpu().tolist())

                num_examples += input_ids.size(0)
                num_steps += 1

        # compute acc, precision, recall and f1
        results.update(acc_and_f1(preds=preds, golds=golds, prefix=split))

        # save predictions and golds
        with open(os.path.join(save_dir, "predictions.txt"), mode="w", encoding="utf-8") as pred_f, \
             open(os.path.join(save_dir, "golds.txt"), mode="w", encoding="utf-8") as gold_f:
            for pred, gold in zip(preds, golds):
                pred_f.write(f"{pred}\n")
                gold_f.write(f"{gold}\n")

    elif args.task == "retrieval":
        all_vectors = []
        all_labels = []
        for batch in eval_bar:
            with torch.no_grad():
                input_ids, pos_input_ids, neg_input_ids, labels = batch

                outputs = model(input_ids, pos_input_ids, neg_input_ids, labels)

                loss = outputs.loss.mean().item()
                loss_list.append(loss)
                vectors = outputs.representation_vectors

                all_vectors.extend(vectors.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

                num_examples += len(vectors)
                num_steps += 1

        # compute map
        all_vectors = np.array(all_vectors)
        all_labels = np.array(all_labels)
        scores = np.matmul(all_vectors, all_vectors.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

        results.update(map_score(scores=scores, sort_ids=sort_ids, labels=all_labels, prefix=split))

        # save predictions
        indices = [example.index for example in raw_examples]
        with open(os.path.join(save_dir, "predictions.jsonl"), mode="w", encoding="utf-8") as pred_f:
            for index, sort_id in zip(indices, sort_ids):
                js = {'index': index, 'answers': []}
                for idx in sort_id[:499]:
                    js['answers'].append(indices[int(idx)])
                pred_f.write(json.dumps(js) + '\n')

    elif args.task == "search":
        all_code_vectors = []
        all_nl_vectors = []
        for batch in eval_bar:
            with torch.no_grad():
                code_input_ids, nl_input_ids = batch

                outputs = model(code_input_ids, nl_input_ids)

                loss = outputs.loss.mean().item()
                loss_list.append(loss)
                code_vectors = outputs.code_vectors
                nl_vectors = outputs.nl_vectors

                all_code_vectors.extend(code_vectors.cpu().tolist())
                all_nl_vectors.extend(nl_vectors.cpu().tolist())

                num_examples += len(code_vectors)
                num_steps += 1

        # compute mrr
        all_code_vectors = np.array(all_code_vectors)
        all_nl_vectors = np.array(all_nl_vectors)
        scores = np.matmul(all_nl_vectors, all_code_vectors.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

        results.update(mrr(scores=scores, prefix=split))

        # save predictions
        indices = []
        urls = []
        for example in raw_examples.examples:
            indices.append(example.idx)
            urls.append(example.url)
        with open(os.path.join(save_dir, "predictions.jsonl"), mode="w", encoding="utf-8") as pred_f:
            for index, url, sort_id in zip(indices, urls, sort_ids):
                js = {'url': url, 'answers': []}
                for idx in sort_id[:100]:
                    js['answers'].append(indices[int(idx)])
                pred_f.write(json.dumps(js) + '\n')

    elif args.task in configs.TASK_TYPE_TO_LIST["seq2seq"]:
        all_preds = []
        all_golds = []
        for batch in eval_bar:
            with torch.no_grad():
                input_ids, labels = batch

                generated_tokens = args.accelerator.unwrap_model(model).generate(
                    input_ids,
                    attention_mask=batch["attention_mask"],
                    **gen_kwargs,
                )

                loss = outputs.loss.mean().item()
                loss_list.append(loss)




    results.update({
        f"{split}_loss": np.mean(loss_list),
        f"{split}_num_examples": num_examples,
        f"{split}_num_steps": num_steps
    })

    # save results
    with open(os.path.join(save_dir, "results.json"), mode="w", encoding="utf-8") as result_f:
        json.dump(results, result_f)

    return results


def run_fine_tune(args):

    logger.info("=" * 20 + " LOADING " + "=" * 20)

    model, tokenizer = build_model_tokenizer(args)

    # watch model
    args.run.watch(model, log_freq=10)

    # prepare data for training and validation
    if args.do_train:
        train_examples, train_dataset, train_dataloader = prepare_data(args, split="train", tokenizer=tokenizer)
    if args.do_valid:
        valid_examples, valid_dataset, valid_dataloader = prepare_data(args, split="valid", tokenizer=tokenizer)

    logger.info(f"Data is loaded and prepared")

    if args.do_train:
        logger.info("=" * 20 + " TRAINING " + "=" * 20)
        # Optimizer
        # Split weights in two groups, one with weight decay and the other not
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

        # Prepare everything with accelerator
        model, optimizer, train_dataloader = args.accelerator.prepare(model, optimizer, train_dataloader)
        if args.do_valid:
            valid_dataloader = args.accelerator.prepare(valid_dataloader)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        else:
            args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Scheduler and math around the number of training steps
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        total_batch_size = args.train_batch_size * args.accelerator.num_processes * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        completed_steps = 0
        early_stop = EarlyStopController(patience=args.patience, higher_is_better=True)

        for epoch in range(args.num_train_epochs):
            model.train()

            train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"[epoch {epoch}, loss x.xxxx]")
            for step, batch in enumerate(train_bar):
                model_kwargs = prepare_model_kwargs(args, batch)

                outputs = model(**model_kwargs)

                loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                args.accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    args.accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                train_bar.set_description(f"[epoch {epoch}, loss {loss.item():.4f}]")
                args.run.log({"train_loss": loss.item(), "epoch": epoch})

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

                if completed_steps >= args.max_train_steps:
                    break

            if args.do_valid:
                torch.cuda.empty_cache()
                logger.info("Start validation")

                results = run_eval(args,
                                   model=model,
                                   tokenizer=tokenizer,
                                   dataloader=valid_dataloader,
                                   raw_examples=valid_examples,
                                   split="valid")
                logger.info(f"End of validation at epoch {epoch}, results:")
                result_table, major_score = postprocess_results(results, major_metric=args.major_metric)
                logger.info(result_table)
                args.run.log(results)

                early_stop(score=major_score, model=model, epoch=epoch)
                args.run.log({"early_stop_counter": early_stop.counter})
                if not early_stop.hit:
                    logger.info(f"Early stopping counter: {early_stop.counter}/{early_stop.patience}")

                # last model
                save_last_dir = os.path.join(args.model_dir, "latest")
                unwrapped_model = args.accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_last_dir, save_function=args.accelerator.save)
                tokenizer.save_pretrained(save_last_dir)
                torch.save(optimizer, save_last_dir)
                logger.info(f"The latest checkpoint is saved to {save_last_dir}")

            if early_stop.early_stop:
                logger.info(f"Early stopping is triggered")
                break

            torch.cuda.empty_cache()

        logger.info("End of training")

        # load and save the best model
        model = early_stop.model
        save_best_dir = os.path.join(args.model_dir, f"best_{args.major_metric}")
        unwrapped_model = args.accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_best_dir, save_function=args.accelerator.save)
        tokenizer.save_pretrained(save_best_dir)
        logger.info(f"The best {args.major_metric} model is saved to {save_best_dir}")

        if args.do_test:
            logger.info("=" * 20 + " TESTING " + "=" * 20)
            torch.cuda.empty_cache()

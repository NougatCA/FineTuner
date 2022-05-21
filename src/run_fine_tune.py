import json
import os
import torch
from torch.optim import AdamW
from transformers import get_scheduler
import logging
import math
from tqdm import tqdm
import numpy as np
from accelerate import Accelerator

from models import build_model_tokenizer, prepare_model_kwargs
from data import prepare_data
import configs
from evaluation.general import acc_and_f1, map_score, mrr, exact_match
from evaluation.google_bleu import google_bleu
from evaluation.smooth_bleu import smooth_bleu
from evaluation.rouge import rouge_l
from evaluation.CodeBLEU.calc_code_bleu import code_bleu
from utils import EarlyStopController, LabelSmoother, postprocess_results

logger = logging.getLogger(__name__)


def run_eval(args, model, tokenizer, dataloader, accelerator: Accelerator, raw_examples, split, epoch=None) -> dict:
    assert split in ["valid", "test"]
    assert epoch is not None or split == "test"
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
                    predictions = accelerator.gather(outputs.predictions)
                    preds.extend(predictions.cpu().tolist())
                else:
                    logits = accelerator.gather(outputs.logits)
                    # logits = outputs.logits
                    pred = np.argmax(logits.cpu().numpy(), axis=1)
                    preds.extend([p.item() for p in pred])

                labels = accelerator.gather(labels)
                golds.extend(labels.squeeze().cpu().tolist())

                num_examples += input_ids.size(0)
                num_steps += 1

        print(preds)
        print(golds)

        # compute acc, precision, recall and f1
        results.update(acc_and_f1(preds=preds, golds=golds, prefix=split))

        # save predictions and golds
        with open(os.path.join(save_dir, "predictions.txt"), mode="w", encoding="utf-8") as pred_f, \
                open(os.path.join(save_dir, "golds.txt"), mode="w", encoding="utf-8") as gold_f:
            for pred, gold in zip(preds, golds):
                pred_f.write(f"{str(pred)}\n")
                gold_f.write(f"{str(gold)}\n")

    elif args.task == "retrieval":
        all_vectors = []
        all_labels = []
        for batch in eval_bar:
            with torch.no_grad():
                input_ids, pos_input_ids, neg_input_ids, labels = batch

                outputs = model(input_ids, pos_input_ids, neg_input_ids, labels)

                loss = outputs.loss.mean().item()
                loss_list.append(loss)
                vectors = accelerator.gather(outputs.representation_vectors)
                labels = accelerator.gather(labels)

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
                code_vectors = accelerator.gather(outputs.code_vectors)
                nl_vectors = accelerator.gather(outputs.nl_vectors)

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
                attention_mask = input_ids.ne(tokenizer.pad_token_id)
                generated_tokens = accelerator.unwrap_model(model).generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_length=args.max_target_length,
                    num_beams=args.num_beams,
                    early_stopping=True
                )
                generated_tokens = accelerator.pad_across_processes(generated_tokens,
                                                                    dim=1,
                                                                    pad_index=tokenizer.pad_token_id)

                generated_tokens = accelerator.gather(generated_tokens).cpu().numpy()
                labels = accelerator.gather(labels).cpu().numpy()

                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_golds = tokenizer.batch_decode(labels, skip_special_tokens=True)

                all_preds.extend([pred.strip() for pred in decoded_preds])
                all_golds.extend([label.strip() for label in decoded_golds])

        # compute bleu, em, rouge-l, etc.
        results.update(exact_match(preds=all_preds, golds=all_golds, prefix=split))
        results.update(google_bleu(preds=all_preds, golds=all_golds, prefix=split))
        results.update(smooth_bleu(preds=all_preds, golds=all_golds, prefix=split))
        results.update(rouge_l(preds=all_preds, golds=all_golds, prefix=split))

        if args.target_lang != "en":
            results.update(code_bleu(preds=all_preds, golds=all_golds, lang=args.target_lang, prefix=split))

        # save predictions and golds
        with open(os.path.join(save_dir, "predictions.txt"), mode="w", encoding="utf-8") as pred_f, \
                open(os.path.join(save_dir, "golds.txt"), mode="w", encoding="utf-8") as gold_f:
            for pred, gold in zip(all_preds, all_golds):
                pred_f.write(pred + "\n")
                gold_f.write(gold + "\n")

    results.update({
        f"{split}_loss": np.mean(loss_list),
        f"{split}_num_examples": num_examples,
        f"{split}_num_steps": num_steps
    })

    # save results
    with open(os.path.join(save_dir, "results.json"), mode="w", encoding="utf-8") as result_f:
        json.dump(results, result_f)

    return results


def run_fine_tune(args, accelerator: Accelerator, run):
    logger.info("=" * 20 + " LOADING " + "=" * 20)

    model, tokenizer = build_model_tokenizer(args)

    # watch model
    run.watch(model, log_freq=10)

    # prepare data for training and validation
    if not args.only_test:
        train_examples, train_dataset, train_dataloader = prepare_data(args, split="train", tokenizer=tokenizer)
        if not args.do_not_valid:
            valid_examples, valid_dataset, valid_dataloader = prepare_data(args, split="valid", tokenizer=tokenizer)

    logger.info(f"Data is loaded and prepared")

    if not args.only_test:
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
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        if not args.do_not_valid:
            valid_dataloader = accelerator.prepare(valid_dataloader)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_epochs * num_update_steps_per_epoch
        else:
            args.num_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

        # Scheduler and math around the number of training steps
        lr_scheduler = get_scheduler(
            name=args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=args.max_train_steps,
        )

        # Label smoothing
        if args.label_smoothing_factor != 0:
            label_smoother = LabelSmoother(epsilon=args.label_smoothing_factor)
        else:
            label_smoother = None

        total_batch_size = args.train_batch_size * args.num_gpus * args.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {args.max_train_steps}")

        completed_steps = 0
        early_stop = EarlyStopController(patience=args.patience, higher_is_better=True)

        for epoch in range(args.num_epochs):
            model.train()

            epoch_losses = []
            train_bar = tqdm(train_dataloader, total=len(train_dataloader), desc=f"[epoch {epoch}, loss x.xxxx]")
            for step, batch in enumerate(train_bar):
                model_kwargs = prepare_model_kwargs(args, batch)

                if label_smoother is not None and \
                        "labels" in model_kwargs and \
                        args.task not in ["retrieval", "search", "cosqa"]:
                    labels = model_kwargs.pop("labels")
                else:
                    labels = None

                outputs = model(**model_kwargs)

                if labels is not None:
                    loss = label_smoother(outputs, labels)
                else:
                    # We don't use .loss here since the model may return tuples instead of ModelOutput.
                    loss = outputs.loss

                # loss = outputs.loss
                loss = loss / args.gradient_accumulation_steps
                accelerator.backward(loss)
                if args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if step % args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    completed_steps += 1

                    epoch_losses.append(loss.item())
                    avg_loss = np.mean(epoch_losses)
                    train_bar.set_description(f"[epoch {epoch}, loss {avg_loss:.4f}]")
                    args.run.log({"train_loss": avg_loss, "epoch": epoch})

                if completed_steps >= args.max_train_steps:
                    break

            if not args.do_not_valid:
                torch.cuda.empty_cache()
                logger.info("Start validation")

                results = run_eval(args,
                                   model=model,
                                   tokenizer=tokenizer,
                                   dataloader=valid_dataloader,
                                   accelerator=accelerator,
                                   raw_examples=valid_examples,
                                   split="valid",
                                   epoch=epoch)
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
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_last_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(save_last_dir)
                torch.save(optimizer, os.path.join(save_last_dir, "optimizer.pt"))
                logger.info(f"The latest checkpoint is saved to {save_last_dir}")

            if early_stop.early_stop:
                logger.info(f"Early stopping is triggered")
                break

            torch.cuda.empty_cache()

        logger.info("End of training")

        # load and save the best model
        model = early_stop.best_model
        save_best_dir = os.path.join(args.model_dir, f"best_{args.major_metric}")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(save_best_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(save_best_dir)
        logger.info(f"The best {args.major_metric} model is saved to {save_best_dir}")

    logger.info("=" * 20 + " TESTING " + "=" * 20)
    torch.cuda.empty_cache()

    # load test data
    logger.info(f"Start loading test data")
    test_examples, test_dataset, test_dataloader = prepare_data(args, split="test", tokenizer=tokenizer)
    test_dataloader = accelerator.prepare_data_loader(test_dataloader)

    test_results = run_eval(args,
                            model=model,
                            tokenizer=tokenizer,
                            dataloader=test_dataloader,
                            accelerator=accelerator,
                            raw_examples=test_examples,
                            split="test")
    logger.info(f"End of testing, results:")
    result_table, _ = postprocess_results(test_results, major_metric=args.major_metric)
    logger.info(result_table)
    args.run.log(test_results)

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
from evaluation.general import acc, p_r_f1, map_score, mrr, exact_match
from evaluation.google_bleu import google_bleu
from evaluation.smooth_bleu import smooth_bleu
from evaluation.rouge import rouge_l
from evaluation.CodeBLEU.calc_code_bleu import code_bleu
from utils import EarlyStopController, LabelSmoother, postprocess_results

logger = logging.getLogger(__name__)


def run_eval(
        args,
        model,
        tokenizer,
        dataloader,
        accelerator: Accelerator,
        run,
        raw_examples,
        split,
        epoch=None) -> dict:
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
        if args.model_type in ["t5", "codet5"] and args.task != "cosqa":
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

                    preds.extend([pred.strip() for pred in decoded_preds])
                    golds.extend([label.strip() for label in decoded_golds])

                    num_examples += input_ids.size(0)
                    num_steps += 1

            # compute acc, precision, recall and f1
            results.update(acc(preds=preds, golds=golds, prefix=split))
            if args.num_labels == 2:
                results.update(p_r_f1(preds=preds, golds=golds, prefix=split, pos_label="true"))
        else:
            for batch in eval_bar:
                with torch.no_grad():
                    if args.task == "cosqa":
                        code_input_ids, nl_input_ids, labels = batch
                        outputs = model(code_input_ids, nl_input_ids, labels)
                    else:
                        input_ids, labels = batch
                        outputs = model(input_ids, labels=labels)

                    loss = outputs.loss
                    if args.num_gpus > 1:
                        loss = loss.mean()
                    loss_list.append(loss.item())

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

            # compute acc, precision, recall and f1
            results.update(acc(preds=preds, golds=golds, prefix=split))
            if args.num_labels == 2:
                results.update(p_r_f1(preds=preds, golds=golds, prefix=split))

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

                loss = outputs.loss
                if args.num_gpus > 1:
                    loss = loss.mean()
                loss_list.append(loss.item())
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

                loss = outputs.loss
                if args.num_gpus > 1:
                    loss = loss.mean()
                loss_list.append(loss.item())
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

                num_examples += input_ids.size(0)
                num_steps += 1

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

    if len(loss_list) > 0:
        results.update({f"{split}_loss": np.mean(loss_list)})
    results.update({
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
    # run.watch(model, log_freq=100)

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
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

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
                    loss = outputs.loss
                # loss = outputs.loss

                if args.num_gpus > 1:
                    loss = loss.mean()
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

                    train_bar.set_description(f"[epoch {epoch}, loss {loss.item():.4f}]")
                    run.log({"train_loss": loss.item(), "epoch": epoch})

                if completed_steps >= args.max_train_steps:
                    break

            if not args.do_not_valid:
                logger.info("Start validation")

                results = run_eval(args,
                                   model=model,
                                   tokenizer=tokenizer,
                                   dataloader=valid_dataloader,
                                   accelerator=accelerator,
                                   run=run,
                                   raw_examples=valid_examples,
                                   split="valid",
                                   epoch=epoch)
                result_table, major_score = postprocess_results(results, major_metric=args.major_metric)
                logger.info(f"End of validation at epoch {epoch}, results:\n{result_table}")
                run.log(results)
                run.log({"valid_major_score": major_score})

                early_stop(score=major_score, model=model, epoch=epoch)
                run.log({"early_stop_counter": early_stop.counter})
                # save the best model
                if early_stop.hit:
                    save_best_dir = os.path.join(args.model_dir, f"best_{args.major_metric}")
                    accelerator.wait_for_everyone()
                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(save_best_dir, save_function=accelerator.save)
                    tokenizer.save_pretrained(save_best_dir)
                    logger.info(f"The best {args.major_metric} model is saved to {save_best_dir}")
                if not early_stop.hit:
                    logger.info(f"Early stopping counter: {early_stop.counter}/{early_stop.patience}")

                # last model
                save_last_dir = os.path.join(args.model_dir, "latest")
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(save_last_dir, save_function=accelerator.save)
                tokenizer.save_pretrained(save_last_dir)
                torch.save(optimizer, os.path.join(save_last_dir, "optimizer.pt"))
                logger.info(f"The latest checkpoint is saved to {save_last_dir}")

                model.train()

            if early_stop.early_stop:
                logger.info(f"Early stopping is triggered")
                break

        logger.info("End of training")

        # load the best model
        best_model_path = os.path.join(args.model_dir, f"best_{args.major_metric}", "pytorch_model.bin")
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.load_state_dict(torch.load(best_model_path))
        logger.info(f"Loaded the best {args.major_metric} model from {best_model_path}")

    logger.info("=" * 20 + " TESTING " + "=" * 20)
    torch.cuda.empty_cache()

    # load test data
    logger.info(f"Start loading test data")
    test_examples, test_dataset, test_dataloader = prepare_data(args, split="test", tokenizer=tokenizer)
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    test_results = run_eval(args,
                            model=model,
                            tokenizer=tokenizer,
                            dataloader=test_dataloader,
                            accelerator=accelerator,
                            run=run,
                            raw_examples=test_examples,
                            split="test")
    result_table, _ = postprocess_results(test_results, major_metric=args.major_metric)
    logger.info(f"End of testing, results:\n{result_table}")
    run.log(test_results)

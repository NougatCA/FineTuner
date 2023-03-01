# FineTuner

## Requirements

The evaluation scripts in [`src/evaluation`](src/evaluation) and the [`src/main.py`](src/main.py) script to fine-tuning and/or start the model's evaluation require the Python packages listed in [`requirements.txt`](requirements.txt) which can be easily installed via `pip install -r requirements.txt`.

### [wandb](https://wandb.ai)

The [`src/main.py`](src/main.py) script requires user's [wandb](https://wandb.ai) key.  Sign up for an account at [https://wandb.ai](https://wandb.ai) and copy your user's key to `wandb_api.key` in the root directory.  In alternative, you could just `export WANDB_API_KEY=___USER_KEY__` before running the [`src/main.py`](src/main.py) script.

## Datasets

All datasets can be downloaded here: [OneDrive](https://1drv.ms/u/s!Aj4XBdlu8BS0gf9b0e1Dze2AkxsqxA?e=p0Whot).
Extract the archive file and put the entire folder in the root directory. 
Or you can put anywhere else and specific the path using the `--data_dir` argument.

## Pre-Trained Models and Tokenizer
All pre-trained models and tokenizer can be downloaded here: [OneDrive](https://1drv.ms/u/s!Aj4XBdlu8BS0gesMNftTjlqQGm64xg?e=Wru6T7)

## Evaluation Scripts

Very easy to use evaluation scripts have been created in `src/evaluation` with detailed comments to refer to.

## Statistical Significance of the results reported in the paper

[OneDrive](https://1drv.ms/x/s!Aj4XBdlu8BS0gf9fQrJtaHOt6q-fKQ?e=S4eE6c)

## Runs

Run `main.py` to start fine-tuning and/or evaluation. 
All arguments are located in `args.py`, specific whatever you need.

Some example scripts are as following.

```shell
# run defect detection task using roberta, with default hyperparameters
python main.py --model roberta --task defect

# run java summarization using codet5
python main.py --model codet5 --task summarization --subset java

# only run evaluation using specific model directory
python main.py --model PATH_TO_MODEL --task clone --only_test

# run code generation using plbart and specific some common arguments
# all gpu devices are used by default, specific device ids by using --cuda_visible_devices, 
# add --no_cuda to disable gpu and use cpu instead
python main.py \
--model plbart \
--task generation \
--num_epochs 10 \
--train_batch_size 64 \
--eval_batch_size 32 \
--max_source_length 64 \
--max_target_length 256 \
--learning_rate 1e-5 \
--num_warmup_steps 1000 \
--cuda_visible_devices 2,3 \
--mixed_precision no # no, fp16, bf16
```

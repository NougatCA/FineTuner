
def get_run_name(args):
    tokens = [args.model, args.task, args.dataset, args.sub_task,
              f"bs{args.batch_size}", f"ep{args.num_epochs}", f"lr{args.learning_rate}", f"warmup{args.warmup_steps}"]
    return "_".join([token for token in tokens if token is not None])

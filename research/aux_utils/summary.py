import csv

from collections import OrderedDict
try: 
    import wandb
except ImportError:
    pass


__all__ = [
    "update_summary"
]


def update_summary(
    epoch: int, 
    train_metrics: OrderedDict, 
    eval_metrics: OrderedDict, 
    filename, 
    write_header=False, 
    log_wandb=False,
    param_hist={},
    **aux_kw 
):
    epoch_summary = OrderedDict(epoch=epoch)
    epoch_summary.update([('train/' + k, v) for k, v in train_metrics.items()])
    epoch_summary.update([('eval/' + k, v) for k, v in eval_metrics.items()])
    epoch_summary.update([(k, v) for k, v in param_hist.items()])
    epoch_summary.update([(k, v) for k, v in aux_kw.items()])

    if log_wandb:
        wandb.log(epoch_summary)
    with open(filename, mode='a') as cf:
        dict_writer = csv.DictWriter(cf, fieldnames=epoch_summary.keys())
        if write_header: 
            dict_writer.writeheader()
        dict_writer.writerow(epoch_summary)

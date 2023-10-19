from sparseml.pytorch.optim import ScheduledModifierManager


def get_current_pruning_modifier(manager : ScheduledModifierManager, epoch: int):
    for pruning_modifier in manager.pruning_modifiers:
        if pruning_modifier.start_epoch <= epoch < pruning_modifier.end_epoch:
            return pruning_modifier
    return None
    

def get_current_learning_rate_modifier(manager : ScheduledModifierManager, epoch: int):
    current_learning_rate_modifier = None
    for lr_modifier in manager.learning_rate_modifiers:
        if lr_modifier.start_epoch <= epoch < lr_modifier.end_epoch:
            current_learning_rate_modifier = lr_modifier
            break
    return current_learning_rate_modifier


def is_update_epoch(manager: ScheduledModifierManager, epoch: int):
    updated = False
    # get current pruning modifier
    pruning_modifier = get_current_pruning_modifier(manager, epoch)
    if pruning_modifier and pruning_modifier.update_frequency > 0:
        updated = (epoch - pruning_modifier.start_epoch) % pruning_modifier.update_frequency == 0
    return updated

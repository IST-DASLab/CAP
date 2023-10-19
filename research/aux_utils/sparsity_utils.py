from torch.nn.parallel import DistributedDataParallel as NativeDDP
# sparseml imports
from sparseml.pytorch.optim import ScheduledModifierManager
from sparseml.pytorch.utils import ModuleSparsificationInfo

from .value_utils import mean_value
from .manager_utils import get_current_pruning_modifier


def get_current_sparsity(manager : ScheduledModifierManager, epoch: int):
    current_pruning_modifier = get_current_pruning_modifier(manager, epoch)
    sparsity = 0.0
    if current_pruning_modifier is not None:
        sparsity = mean_value(current_pruning_modifier.applied_sparsity)
    return sparsity


def get_sparsity_info(model):
    if isinstance(model, NativeDDP):
        model = model.module
    sparsity_info = ModuleSparsificationInfo(model)
    return str(sparsity_info)
    
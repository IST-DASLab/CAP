# Copyright (c) 2021 - present / Neuralmagic, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Modifiers and support for structured (channel/filter) pruning
Thinning (removal of pruned channels) implemented by LayerThinningModifier
"""
import math
import torch

from torch import Tensor
from torch.nn import Parameter, Module
from typing import Any, Dict, List, Optional, Union

from sparseml.pytorch.utils import GradSampler
from sparseml.pytorch.utils.logger import BaseLogger
from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from sparseml.pytorch.sparsification.pruning.mask_creator import (
    PruningMaskCreator,
    get_mask_creator_default,
)
from sparseml.pytorch.sparsification.pruning.modifier_pruning_magnitude import (
    GMPruningModifier,
)
from sparseml.pytorch.sparsification.pruning.scorer import PruningParamsGradScorer


__all__ = [
    "MovementPruningModifier",
    "MovementPruningParamsScorer",
]


@PyTorchModifierYAML()
class MovementPruningModifier(GMPruningModifier):
    """
    Gradually applies kernel sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given amount of time
    and applied with an interpolated function for each step taken.

    Uses movement pruning to gradually mask parameter values.
    Movement pruning introduced here: https://arxiv.org/abs/2005.07683
    Pruning is unstructured by default, structure can be specified by mask_type.

    | Sample yaml:
    |   !MovementPruningModifier
    |       init_sparsity: 0.05
    |       final_sparsity: 0.8
    |       start_epoch: 0.0
    |       end_epoch: 10.0
    |       update_frequency: 1.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       mask_type: unstructured
    |       num_grads: 1024

    :param init_sparsity: the initial sparsity for the param to start with at
        start_epoch
    :param final_sparsity: the final sparsity for the param to end with at end_epoch.
        Can also be a Dict of final sparsity values to a list of parameters to apply
        them to. If given a Dict, then params must be set to [] and the params to
        be pruned will be read from the final_sparsity Dict
    :param start_epoch: The epoch to start the modifier at
    :param end_epoch: The epoch to end the modifier at
    :param update_frequency: The number of epochs or fraction of epochs to update at
        between start and end
    :param params: A list of full parameter names or regex patterns of names to apply
        pruning to.  Regex patterns must be specified with the prefix 're:'. __ALL__
        will match to all parameters. __ALL_PRUNABLE__ will match to all ConvNd
        and Linear layers' weights. If a sparsity to param mapping is defined by
        final_sparsity, then params should be set to []
    :param leave_enabled: True to continue masking the weights after end_epoch,
        False to stop masking. Should be set to False if exporting the result
        immediately after or doing some other prune
    :param inter_func: the type of interpolation function to use:
        [linear, cubic, inverse_cubic]
    :param mask_type: String to define type of sparsity (options: ['unstructured',
        'block']), List to define block shape of a parameters in and out
        channels, or a SparsityMaskCreator object. default is 'unstructured'
    """

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: Union[float, Dict[float, List[str]]],
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        mask_type: str = "unstructured",
        num_grads: int = 1024,
        grad_sampler_kwargs: Dict[str, Any] = {},
        **kwargs
    ):
        super(MovementPruningModifier, self).__init__(
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            params=params,
            leave_enabled=leave_enabled,
            inter_func=inter_func,
            mask_type=mask_type,
        )

        self._num_grads = num_grads
        self._grad_sampler_kwargs = grad_sampler_kwargs

    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        return get_mask_creator_default(self.mask_type)

    def _get_scorer(self, params: List[Parameter]) -> PruningParamsGradScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """
        return MovementPruningParamsScorer(params=params)

    @ModifierProp(serializable=False)
    def global_sparsity(self) -> bool:
        """
        :return: True for global magnitude pruning, False for
            layer-wise. [DEPRECATED] - use GlobalMagnitudePruningModifier
            for global magnitude pruning and MagnitudePruningModifier for layer-wise
        """
        return self._global_sparsity

    def initialize(
        self,
        module: Module,
        epoch: float = 0,
        loggers: Optional[List[BaseLogger]] = None,
        **kwargs,
    ):
        """
        Grab the layers and apply if epoch in range to control pruning for.
        Expects `grad_sampler` dict with `data_loader_builder` and `loss_function`
        to initialize GradSampler instance and optionally override data-loader's
        hyperparams with `grad_sampler_kwargs` given in the recipe.

        :param module: the PyTorch model/module to modify
        :param epoch: the epoch to initialize the modifier and module at.
            Defaults to 0 (start of the training process)
        :param loggers: optional list of loggers to log the modification process to
        :param kwargs: optional kwargs to support specific arguments
            for individual modifiers.
        """
        if (
            "grad_sampler" not in kwargs
            or "data_loader_builder" not in kwargs["grad_sampler"]
            or "loss_fn" not in kwargs["grad_sampler"]
        ):
            raise RuntimeError(
                "grad_sampler dict with data_loader_builder and loss_fn "
                "must be provided to initialize GradSampler"
            )

        self._grad_sampler = GradSampler(
            kwargs["grad_sampler"]["data_loader_builder"](
                **self._grad_sampler_kwargs
            ),
            kwargs["grad_sampler"]["loss_fn"],
        )

        super().initialize(module, epoch, loggers, **kwargs)


    def _collect_grad_samples(
        self,
        module: Module,
        grad_sampler: GradSampler,
    ):
        if not isinstance(grad_sampler, GradSampler):
            raise ValueError(
                "One-shot OBS pruning requires a GradSampler object given by the "
                f"grad_sampler kwarg. Given an object of type {type(grad_sampler)}"
            )

        is_training = module.training
        module.eval()

        for _ in grad_sampler.iter_module_backwards(module, self._num_grads):
            self._module_masks.pre_optim_step_update()

        if is_training:
            module.train()


    def _prepare(self,  module: Module):
        # clear cache in the beginning and end of training
        torch.cuda.empty_cache()
        if self._scorer._is_main_proc:
            self._scorer._enabled_grad_buffering = True

        # collect grads for empirical inverse Fisher estimation
        self._collect_grad_samples(module, self._grad_sampler)
        self._pre_step_completed = True

        if self._scorer._is_main_proc:
            self._scorer._enabled_grad_buffering = False
        torch.cuda.empty_cache()
    

    def check_mask_update(
        self, module: Module, epoch: float, steps_per_epoch: int, **kwargs
    ):
        """
        Update mask values if necessary

        :param module: module to modify
        :param epoch: current epoch and progress within the current epoch
        :param steps_per_epoch: number of steps taken within each epoch
            (calculate batch number using this and epoch)
        """
        started = self.started
        if self.start_pending(epoch, steps_per_epoch):
            self._module_masks.enabled = True
            started = True

        if not self._pre_step_completed:
            # do pre optim step before mask update on update steps
            self._module_masks.pre_optim_step_update()
            self._pre_step_completed = True

        if started:
            # get sparsity level to be applied
            self._applied_sparsity = self.get_applied_sparsity_for_epoch(
                epoch, steps_per_epoch
            )

            self._prepare(module)

            self._module_masks.update_param_masks(target=self._applied_sparsity)
            self._sparsity_applied = True

        if self.end_pending(epoch, steps_per_epoch):
            self._module_masks.pruning_end(self._leave_enabled)


class MovementPruningParamsScorer(PruningParamsGradScorer):
    """
    Scores parameters based on their movement which is defined as
    movement_score = sum(-1.0 * W * dL/dW)

    Movement pruning introduced here: https://arxiv.org/abs/2005.07683

    :param params: list of model Parameters to track and score
    """

    def __init__(self, params: List[Parameter]):
        super().__init__(params)
        self._movement_scores = [torch.zeros_like(param) for param in self._params]


    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored by their weight times the direction
            of their gradient.
        """
        if self._is_main_proc:
            for i, score in enumerate(self._movement_scores):
                score[self._masks[i] == 0] = float("-inf")

        self._broadcast_list_from_main(self._movement_scores)

        # return self._movement_scores
        return self._movement_scores
        

    def pre_optim_step_update(self, masks: List[Tensor]):
        """
        Update movement scores based on the current Parameter weights and gradients

        :param masks: latest masks that are applied to these parameters
        """
        self._masks = masks  # to be used by score_parameters

        for idx, param in enumerate(self._params):
            if param.grad is not None and not torch.any(param.grad.isnan()):
                self._movement_scores[idx].add_(-param.grad * param.data)


    def mask_update(self, masks: List[Tensor], mask_diffs: List[Tensor]):
        """
        Resets non main process scores after they have been recorded in the main
        process during the mask update

        :param masks: latest masks to be applied to these parameters
        :param mask_diffs: mask diff values returned by mask_difference for these
            masks that describe how these masks changed since the last update
        """
        if not self._is_main_proc:
            for score in self._movement_scores:
                score *= 0.0

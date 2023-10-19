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
Modifier classes implementing the pruner 
from https://arxiv.org/abs/2210.09223.
"""
import os
import math
import torch
import logging
import numpy as np

from torch import Tensor
from torch.nn import Module, Parameter
from typing import Any, Dict, List, Optional, Union

from sparseml.pytorch.utils import tensor_sparsity
from sparseml.pytorch.sparsification.modifier import ModifierProp, PyTorchModifierYAML
from .mask_creator import  PruningMaskCreator, get_mask_creator_default
from .modifier_pruning_obs import OBSPruningModifier
from .scorer import PruningParamsGradScorer
from .pruning_handle import CAPHandle, CAPNMHandle
from .modifier_pruning_obs import EmpiricalBlockFisherInverse


__all__ = [
    "CAPruningModifier",
    "CAPruningParamsScorer",
]


_LOGGER = logging.getLogger(__name__)


@PyTorchModifierYAML()
class CAPruningModifier(OBSPruningModifier):
    """
    As described in https://arxiv.org/abs/2210.09223.

    Gradually applies sparsity to a given parameter or parameters from
    init_sparsity until final_sparsity is reached over a given number of epochs.
    When pruning, it also updates remaining weights to compensate for accuracy drops incurred
    by pruning. It follows the Optimal Brain Surgeon framework with approximations
    and optimizations to make it efficient but accurate for large models.

    Naming convention with respect to the paper:
        * damp == small dampening constant 'lambda'
        * num_grads == number of gradient outer products 'm'
        * fisher_block_size == size of the blocks 'B' along the main diagonal

    Memory requirements: O(dB), where 'd' is the total number of prunable weights.
    If O(dB) can't fit on a single GPU device, pytorch DDP should be used to split
    the computational overhead equally between devices.

    Supported mask types: unstructured and block4.

    | Sample yaml:
    |   !CAPruningModifier
    |       init_sparsity: 0.7
    |       final_sparsity: 0.9
    |       start_epoch: 2.0
    |       end_epoch: 26.0
    |       update_frequency: 4.0
    |       params: ["re:.*weight"]
    |       leave_enabled: True
    |       inter_func: cubic
    |       global_sparsity: True
    |       mask_type: unstructured
    |       num_grads: 1024
    |       damp: 1e-7
    |       fisher_block_size: 50
    |       grad_sampler_kwargs:
    |           batch_size: 8

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
    :param mask_type: String to define type of sparsity to apply. 'unstructured'
        'block4', 'N:M are supported. Default is 'unstructured'. For N:M provide
        two integers that will be parsed. 
    :param global_sparsity: set True to enable global pruning. If False, pruning will
        be layer-wise. Default is True
    :param num_grads: number of gradients used to calculate the Fisher approximation
    :param damp: dampening factor, default is 1e-7
    :param fisher_block_size: size of blocks along the main diagonal of the Fisher
        approximation, default is 50
    :param grad_sampler_kwargs: kwargs to override default train dataloader config
        for pruner's gradient sampling.
    :param num_recomputations: number of EmpiricalFisher matrix recomputations
    :param blocks_in_parallel: amount of rows traversed simultaneously by CAP pruning modifier
    :param fisher_inv_device: select specific device to store Fisher inverses.
    :param traces_backup_dir: str. If one would like to store pruning traces on disk, one can 
        specify temporary dir for storage. 
    """

    _supported_masks = ("unstructured", "N:M")

    def __init__(
        self,
        init_sparsity: float,
        final_sparsity: float,
        start_epoch: float,
        end_epoch: float,
        update_frequency: float,
        params: Union[str, List[str]],
        leave_enabled: bool = True,
        inter_func: str = "cubic",
        global_sparsity: bool = True,
        mask_type: str = "unstructured",
        num_grads: int = 1024,
        damp: float = 1e-7,
        fisher_block_size: int = 50,
        fisher_inv_device: Optional[str] = None,
        blocks_in_parallel: int = -1,
        grad_sampler_kwargs: Dict[str, Any] = {},
        num_recomputations: int = 1,
        recomputation_inter_func: str = "linear",
        # offloading params
        traces_backup_dir: Optional[str] = None
    ):
        # setup N:M sparsity
        self._n = None
        self._m = None
        # validation of N:M args
        if ':' in mask_type:
            assert len(mask_type.split(':')) == 2, "Expected format N:M"
            n, m = [int(x) for x in mask_type.split(':')]
            # no need for additional recomputations for N:M sparsity
            assert num_recomputations == 1
            assert init_sparsity == final_sparsity == (n / m)
            self._n = n
            self._m = m
            # to process afterwards is usual
            mask_type = "N:M"

        super().__init__(
            params=params,
            init_sparsity=init_sparsity,
            final_sparsity=final_sparsity,
            inter_func=inter_func,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            update_frequency=update_frequency,
            global_sparsity=global_sparsity,
            leave_enabled=leave_enabled,
            mask_type=(mask_type, "unstructured")[mask_type=='N:M'],
            fisher_block_size=fisher_block_size,
            fisher_inv_device=fisher_inv_device,
            damp=damp,
            num_grads=num_grads,
            grad_sampler_kwargs=grad_sampler_kwargs,
            num_recomputations=num_recomputations,
            recomputation_inter_func=recomputation_inter_func,
        )
        
        self._traces_backup_dir = traces_backup_dir  
        self._blocks_in_parallel = blocks_in_parallel  


    @ModifierProp()
    def n(self) -> int:
        """
        :return: n in N:M sparsity
        """
        return self._n

    @ModifierProp()
    def m(self) -> int:
        """
        :return: n in N:M sparsity
        """
        return self._m


    def _get_mask_creator(
        self, param_names: List[str], params: List[Parameter]
    ) -> PruningMaskCreator:
        """
        :param names: full names of parameters to be pruned
        :param params: list of Parameters to be masked
        :return: mask creator object to be used by this pruning algorithm
        """
        # we use unstructured even for N:M pattern
        return get_mask_creator_default("unstructured")


    def _get_scorer(self, params: List[Parameter]) -> PruningParamsGradScorer:
        """
        :param params: list of Parameters for scorer to track
        :return: param scorer object to be used by this pruning algorithm
        """

        return CAPruningParamsScorer(
            params=params,
            num_grads=self._num_grads,
            damp=self._damp,
            fisher_block_size=self._fisher_block_size,
            mask_type=self._mask_type,
            blocks_in_parallel=self._blocks_in_parallel,
            fisher_inv_device=self._fisher_inv_device,
            traces_backup_dir=self._traces_backup_dir
        )

    
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


class CAPruningParamsScorer(PruningParamsGradScorer):
    """
    Scores parameters using the equations introduced in the Optimal BERT Surgeon
    to solve for the optimal weight update in the Optimal Brain Surgeon (OBS)
    framework. Implements unstructured and semi-structured (block4) scoring and
    pruning.

    :param params: list of model Parameters to track and score
    :param num_grads: number of gradients used to calculate the Fisher approximation
    :param damp: dampening factor, default is 1e-7
    :param fisher_block_size: size of blocks along the main diagonal of the Fisher
        approximation, default is 50
    """

    def __init__(
        self,
        params: List[Parameter],
        num_grads: int,
        damp: float,
        fisher_block_size: int,
        mask_type: str,
        n: int = 2,
        m: int = 4,
        blocks_in_parallel: int = -1,
        fisher_inv_device: Optional[str] = None,
        traces_backup_dir: Optional[str] = None,
    ):
        super().__init__(params)
        self._damp = damp
        self._num_grads = num_grads
        self._fisher_block_size = fisher_block_size
        self._mask_type = mask_type

        self._Finvs: List[EmpiricalBlockFisherInverse] = None
        self._enabled_grad_buffering = False
        self._eps = torch.finfo(torch.float32).eps

        # assign device to each Finv
        self._devices = []
        num_devices = torch.cuda.device_count()
        if fisher_inv_device:
            self._devices = [torch.device(fisher_inv_device)] * len(self._params)
            num_devices = 1
        elif num_devices == 0:
            self._devices = [torch.device("cpu")] * len(self._params)
        else:
            num_devices = min(num_devices, len(self._params))
            per_device = math.floor(len(self._params) / num_devices)
            for i in range(num_devices):
                self._devices += [torch.device("cuda", i)] * per_device
            remainder = len(self._params) - len(self._devices)
            if remainder > 0:
                self._devices += [self._devices[-1]] * remainder
        self._num_devices = max(num_devices, 1)

        self._pickle_exclude_params.extend(
            [
                "_Finvs",
                "_enabled_grad_buffering",
                "_devices",
            ]
        )

        if traces_backup_dir:
            os.makedirs(traces_backup_dir, exist_ok=True)

        # init pruning handles
        self._handles: List[Union[CAPHandle, CAPHandle]] = [None] * len(params)
        for i, param in enumerate(params):
            # add obs handle to each module
            if mask_type == 'unstructured':
                self._handles[i] = CAPHandle(
                    param,
                    blocks_in_parallel=blocks_in_parallel,
                    verbose=False,
                    backup_path=(os.path.join(traces_backup_dir, f'{i}.pth') if traces_backup_dir else None),
                    device=self._devices[i]
                )
            elif mask_type == 'N:M':
                self._handles[i] = CAPHandle(
                    param,
                    n=n,
                    m=m,
                    blocks_in_parallel=blocks_in_parallel,
                    verbose=False,
                    backup_path=None,
                    device=self._devices[i]
                )


    def _setup_FisherInverse(self, masks: List[Tensor]):

        def divisor_generator(n):
            large_divisors = []
            for i in range(1, int(math.sqrt(n) + 1)):
                if n % i == 0:
                    yield i
                    if i * i != n:
                        large_divisors.append(n // i)
            for divisor in reversed(large_divisors):
                yield divisor


        self._masks = masks  # to be used by score_parameters
        self._Finvs = []
        for i, param in enumerate(self._params):
            # get numer
            divisors = np.array(list(divisor_generator(param.numel())))
            adj_fisher_block_size = \
                divisors[np.searchsorted(divisors, self._fisher_block_size)]
            self._Finvs.append(
                EmpiricalBlockFisherInverse(
                    self._num_grads,
                    adj_fisher_block_size,
                    param.numel(),
                    self._damp,
                    self._devices[i],
                )
            )


    @torch.no_grad()
    def score_parameters(self) -> List[Tensor]:
        """
        :return: List of Tensors the same shapes as the given Parameters where
            each Parameter's elements are scored based on the blockwise OBS
        """
        scores = [None] * len(self._params)

        if self._is_main_proc:
            for i, obc_handle in enumerate(self._handles):
                # set fisher inverse
                obc_handle.set_Finv(self._Finvs[i].F_inv)
                # compute losses and weight traces
                obc_handle.run()
                scores[i] = obc_handle.losses.reshape(obc_handle.shape_orig)
                # scores are losses
                scores[i][self._masks[i] == 0] = float("-inf")

        self._broadcast_list_from_main(scores)

        return scores

    @torch.no_grad()
    def pre_optim_step_update(self, masks: List[Tensor]):
        """
        Update the empirical inverse Fisher estimation based on the current gradients

        :param masks: latest masks that are applied to these parameters
        """
        if not self._enabled_grad_buffering:
            # only collect gradients when called during pruning step
            # this ignores calls invoked by manager during training
            return

        if self._Finvs is None:
            self._setup_FisherInverse(masks)

        for i, finv in enumerate(self._Finvs):
            self._params[i].grad.mul_(masks[i])
            finv.add_grad(self._params[i].grad.view(-1).to(self._devices[i]))


    @torch.no_grad()
    def mask_update(self, masks: List[Tensor], mask_diffs: List[Tensor]):
        """
        Apply OBS weight update which zeros-out pruned weights and updates the
        remaining weights to preserve the loss.

        :param masks: latest masks to be applied to these parameters
        :param mask_diffs: mask diff values returned by mask_difference for these
            masks that describe how these masks changed since the last update
        """

        obc_weights = [None] * len(self._params)
        if self._is_main_proc:
            for i, obc_handle  in enumerate(self._handles):
                if self._mask_type == 'unstructured':
                    param_sparsity = tensor_sparsity(masks[i])
                    # get weight from the mask sparsity and pruning traces
                    obc_weight = obc_handle.get_pruning_database([param_sparsity])[0]
                    # update weight in obc_handle
                    obc_handle.W  = obc_weight
                # for N:M sparsity just take the weight
                else:
                    obc_weight = obc_handle.W
                obc_weights[i] = obc_weight

        self._broadcast_list_from_main(obc_weights)
        # set weight according to the OBC selection
        for i, param in enumerate(self._params):
            param.data = obc_weights[i].to(param.data.device)

        self._Finvs = None
        
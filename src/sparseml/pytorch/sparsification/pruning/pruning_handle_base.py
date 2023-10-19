import os
import shutil
import numpy as np

from torch.nn import Module
from typing import List, Any, Optional
from abc import ABC, abstractmethod


__all__ = [
    "BasePruningHandle"
]


class BasePruningHandle(ABC):

    '''
    :param layer: Layer for which the reconstruction 
        database is constructed.
    :param storage_dir: Directory where the data related to the pruning 
        of the given layer is stored.
    '''

    def __init__(
        self,
        layer: Module,
        storage_dir: Optional[str] = None,
    ) -> None:
        self._layer = layer
        self._sparsity_levels = []
        self._storage_dir = storage_dir
        self._weight = layer.weight
        self._weight_shape = layer.weight.shape
        self._mask = None
        # weight properties
        self._dim_out = self._weight.shape[0]
        self._dim_in  = np.prod(self._weight.shape[1:])
        # device where the weight was originally stored
        self._orig_device = self._weight.device
        # flag indicating whether one can query for weight
        self._is_built = False
        # weight reconstruction database
        self._reconstruction_database = {}

        if self._storage_dir is not None and len(self._storage_dir) > 0:
            os.makedirs(self._storage_dir, exist_ok=True)

    @property
    def is_built(self):
        return self._is_built

    @property
    def weight(self):
        return self._weight

    @abstractmethod
    def prepare(self, *args, **kwargs):
        self._is_built = True

    @abstractmethod
    def get_reconstruction_loss(self, sparsity: float):
        pass

    @abstractmethod
    def build(self, sparsity_levels: Optional[List[float]] = None):
        '''
        :param sparsity_levels: Optional[List] of sparsity levels for which to 
        construct and store versions of weights for a given layer. 
        If not provided assumes that the weight with queried sparsity may
        be constructed on fly.
        '''
        if sparsity_levels is not None:
            assert sparsity_levels == sorted(sparsity_levels), \
                "Sparsity levels have to be provided in ascending order"

    def set(self, sparsity: float) -> float:
        '''
        :param sparsity: queried sparsity
        '''
        if not self.is_built:
            raise RuntimeError("Not prepared for querying. Run prepare() and build() first.")

    @abstractmethod
    def free(self):
        if self._storage_dir:
            shutil.rmtree(self._storage_dir)

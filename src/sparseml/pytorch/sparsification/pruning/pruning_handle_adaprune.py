

class AdaPruneReconstructionDatabase(BaseReconstructionDatabase):

    def __init__(
        self, layer: Module, 
        sparsity_levels: List[float], 
        storage_dir: Optional[str],
        # AdaPrune params
        calibration_steps: int,
        calibration_batch_size: int,
        calibration_lr: float = 0.1,
        calibration_momentum: float = 0.9,
    ) -> None:
        super().__init__(layer, sparsity_levels, storage_dir)
        # AdaPrune calibration params
        self._calibration_steps = calibration_steps
        self._calibration_batch_size = calibration_batch_size
        self._calibration_lr = calibration_lr
        self._calibration_momentum = calibration_momentum
        # cached inputs and outputs for given layer
        self._cached_inputs  = None
        self._cached_outputs = None 

    def cache_inputs(self, layer_inputs, layer_outputs):
        pass

    def prepare(self):
        pass

    def _get_score(self):
        return torch.abs(self._weight)

    def _optimize(self):
        # create calibration optimizer
        optimizer = SGD(
            [self._weight], 
            lr=self._calibration_lr, 
            momentum=self._calibration_momentum
        )
        # run calibration
        for step in range(self._calibration_steps):
            # sample random ids
            batch_ids = torch.randperm(len(self._cached_inputs))[:self._calibration_batch_size]
            # get batch of layer inputs and outputs
            layer_inputs = self._cached_inputs[batch_ids]
            layer_outputs = self._cached_outputs[batch_ids]
            # make optimizer step
            optimizer.zero_grad()
            pred_outputs = self._layer(layer_inputs)
            loss = F.mse_loss(pred_outputs, layer_outputs)
            loss.backward()
            optimizer.step()
            # mask weight to prevent update of zeroed weights
            with torch.no_grad():
                self._weight.data *= self._mask


    def build(self):
        # create optimizer
        for sparsity_level in self._sparsity_levels:
            # compute scores
            score = self._get_score()
            # compute sparsification threshold
            threshold = torch.kthvalue(
                score.view(-1), k=int(sparsity_level * self._weight.numel())
            )
            # get mask and prune to current sparsity level
            self._mask = score <= threshold
            with torch.no_grad():
                self._weight *= self._mask
            # optimize with current sparsity level
            self._optimize()


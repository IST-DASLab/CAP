### Official repository of CAP
---

To facilitate reproducibility of our experiments, we integrate our pruners with the popular open-source library [SparseML](https://github.com/neuralmagic/sparseml) and build on top of the `rwightman`'s `train.py` script from [https://github.com/rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

### Structure of the repository
---

The modified source code from `SparseML` is located in `src/` and its subdirectories. Main pruning algorithms are implemented in `src/sparseml/pytorch/sparsification/pruning` directory as SparseML PruningModifiers. Notably:

- CAPruningModifier: `modifier_pruning_cap.py` (our CAP pruner)
- FastCAPruningModifier: `modifier_pruning_fast_cap.py` (our FastCAP pruner)
- GlobalMagnitudePruningModifier: `modifier_pruning_magnitude.py` (implemented by [NeuralMagic](https://neuralmagic.com/))
- OBSPruningModifier: `modifier_pruning_obs.py` (implemented by authors of the [oBERT paper](https://arxiv.org/abs/2203.07259)) 

The code to launch experiments is located inside `research/` directory. 

- ```research/``` — root directory for experiments \
    ```├── sparse_training.py``` — main script for gradual pruning (based on [train.py](https://github.com/rwightman/pytorch-image-models/blob/master/train.py) from timm) \
    ```├── one_shot_pruning.py``` — script for running one-shot pruning experiments \
    ```├── run_gradual_pruning.sh``` — script to launch `sparse_training.py` \
    ```├── run_one_shot_pruning.sh``` — script to launch `one_shot_pruning.py` \
    ```├── utils/``` — additional utils used in training scripts \
    ```├── configs/``` — `.yaml` recipes with training hyperparameters \
    ```├── recipes/``` — SparseML recipes for pruning


### Usage
---

**Installation**

The recommended way to run `CAP` is via conda enviroment.

**Configure enviroment**

One needs to install `torch` with GPU support and `timm` library to run the code:

Follow the steps below to setup a conda environment:
```bash
conda create --name CAP python==3.9
conda activate CAP
conda install scipy numpy scikit-learn pytorch=1.13.1 torchvision==0.14.1 torchaudio==0.13.1 cudatoolkit=11.3 -c pytorch 
pip install -r requirements.txt
```

To install `SparseML` type (in the root directory of the project):
```
python setup.py install
```

(*Optional*) We use [W&B](https://wandb.ai) for logging. Install it via `pip` in case you want to log data there:

```
pip install wandb
```

If logging to W&B  prior to launching script define W&B environment variables:
```bash
export WANDB_ENTITY=<your_entity>
export WANDB_PROJECT=<project_name>
export WANDB_NAME=<run_name>
```

**Workflow**

- Select a config with training hyperparameters (`research/configs`)
- Select a SparseML recipe (`research/recipes`)
- Define other hyperparams in the launch script (`research/run_gradual_pruning.sh` or `research/run_one_shot_pruning.sh`)
- Enjoy!

**Example usage**

Recipes used in the paper are located in `research/recipes` directory. 
Choose a recipe from `one_shot` subdirectory for one-pruning and `one_shot+finetune`
for one-shot+finetune pruning and  `gradual_pruning` for experiments with a gradual increase of sparsity level.

**One-shot pruning**

```bash
python one_shot_pruning.py \
    \
    --data-dir <data_dir> \
    \
    --sparseml-recipe <path_to_recipe> \
    \
    --model <model_name> \
    \
    --experiment <experiment_name> \
    \
    -gb <gs_loader_batch_size> \
    -vb <validation_batch_size> \
    \
    --sparsities <list_of_sparsities>
```

**One-shot+finetune/gradual pruning**

```bash
python -m torch.distributed.launch \
    --nproc_per_node=<num_proc> \
    --master_port=<master_port> \
    sparse_training.py \
    \
    --data-dir <data_dir> \
    \
    --sparseml-recipe <path_to_recipe> \
    \
    --model <model_name> \
    \
    --experiment <experiment_name> \
    \
    -gb <gs_loader_batch_size> \
    -vb <validation_batch_size> \
    \
    --sparsities <list_of_sparsities>
```

**Tweaking CAP hyperparameters**

There are several hyperparameters in the oViT method that can be adjusted for better peformance and tuned for each model/dataset. We provide some defaults that should work well across many different models, as demonstrated in the paper.

```
    :param mask_type: String to define type of sparsity to apply. 'unstructured'
        'block4', 'N:M' are supported. Default is 'unstructured'. For N:M provide
        two integers that will be parsed, e.g. '2:4'
    :param num_grads: number of gradients used to calculate the Fisher approximation
    :param damp: dampening factor, default is 1e-7
    :param fisher_block_size: size of blocks along the main diagonal of the Fisher
        approximation, default is 50
    :param grad_sampler_kwargs: kwargs to override default train dataloader config
        for pruner's gradient sampling
    :param num_recomputations: number of EmpiricalFisher matrix recomputations
    :param blocks_in_parallel: amount of rows traversed simultaneously by OBSX pruning modifier
    :param fisher_inv_device: select specific device to store Fisher inverses.
    :param traces_backup_dir: str. If one would like to store pruning traces on disk, one can 
        specify temporary dir for storage. 
```

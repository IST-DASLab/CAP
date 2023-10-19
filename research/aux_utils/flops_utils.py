LAYER_NORM_FLOPS = 5
# GELU: 0.5 * x * (1 + tanh(sqrt(2 / np.pi) * (x + 0.044715 * pow(x, 3))))
ACTIVATION_FLOPS = 8
# max/substract (for stability), exp, sum, divide
SOFTMAX_FLOPS = 5

def compute_flops_ViT(
    sparsity_distribution: dict, 
    D: int, # embed_dim
    L: int, # num tokens
    num_blocks: int = 12,
    image_size: int = 224,
    patch_size: int = 16,
    num_classes: int = 1000
):
    sparse_flops, dense_flops = 0, 0
    # embed proj
    dense_flops  += 3 * D * image_size ** 2
    sparse_flops += 3 * D * image_size ** 2
    # iterate over all blocks in the model
    for block_id in range(num_blocks):
        # compute attention FLOPs
        if sparsity_distribution.get(f'blocks.{block_id}.attn.qkv.weight'):
            qkv_sparsity = sparsity_distribution[f'blocks.{block_id}.attn.qkv.weight']['sparsity']
        else:
            qkv_sparsity = sum(
                sparsity_distribution[f'blocks.{block_id}.attn.{x}.weight']['sparsity'] 
                for x in ['q', 'k', 'v']
            ) / 3.0
        proj_sparsity = sparsity_distribution[f'blocks.{block_id}.attn.proj.weight']['sparsity']
        # qkv flops
        dense_flops  += (3 * D ** 2 * L + 3 * D * L)
        sparse_flops += (3 * (1 - qkv_sparsity) * D ** 2 * L + 3 * D * L)
        # Q K^T flops
        dense_flops  += D * L ** 2
        sparse_flops += D * L ** 2
        # softmax
        dense_flops  += SOFTMAX_FLOPS * L ** 2
        sparse_flops += SOFTMAX_FLOPS * L ** 2
        # proj flops
        dense_flops  +=  (D * L ** 2 + D * L)
        sparse_flops += ((1 - proj_sparsity) * D * L ** 2 + D * L)
        # compute MLP FLOPs
        fc1_sparsity = sparsity_distribution[f'blocks.{block_id}.mlp.fc1.weight']['sparsity']
        fc2_sparsity = sparsity_distribution[f'blocks.{block_id}.mlp.fc2.weight']['sparsity']
        dense_flops  += (8 * D ** 2 *  L + 8 * D * L)
        sparse_flops += (4 * (2 - fc1_sparsity - fc2_sparsity) * D ** 2 * L + 8 * D * L)
        # Layer Norm FLOPs
        dense_flops  += 2 * LAYER_NORM_FLOPS * L * D
        sparse_flops += 2 * LAYER_NORM_FLOPS * L * D
        # Activation FLOPS
        dense_flops  += 4 * ACTIVATION_FLOPS * L * D
        sparse_flops += 4 * ACTIVATION_FLOPS * L * D
        # Skip connection FLOPs
        dense_flops += 2 * L * D
        sparse_flops += 2 * L * D

    # head FLOPS
    dense_flops += D * num_classes
    sparse_flops += D * num_classes
    return int(dense_flops), int(sparse_flops)

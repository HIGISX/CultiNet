# CultiNet Model

This directory contains the model and training source:

- `run.py`: training and evaluation entry point.
- `options.py`: command-line configuration.
- `train.py`: training loop and evaluation routines.
- `nets/`: facility encoder, cached attention decoder, and REINFORCE baselines.
- `problems/`: problem definition and decoding state.
- `utils/`: data, tensor, and logging utilities.

## Dependencies

```bash
python -m pip install torch numpy Pillow scipy tqdm tensorboard_logger
```

Select the appropriate PyTorch build for your CPU or CUDA environment.

## Data interface

Each dataset is a pickled list of dictionaries containing `users`, `facilities`, `demand`, `p`, and `r`. After batching, the tensors have the following shapes:

| Key | Shape | Description |
| --- | --- | --- |
| `users` | `[B, N, 2]` | Demand-point coordinates |
| `facilities` | `[B, M, 5]` | Candidate features; the first two columns are coordinates |
| `demand` | `[B, N, 1]` | Demand weights |
| `p` | `[B]` | Number of facilities to select |
| `r` | `[B]` | Service radius in the same units as the coordinates |

`B` is the batch size, `N` the number of demand points, and `M` the number of candidates. Use floating-point coordinates, features, and demand weights. Set `p` and `r` consistently within each batch, with `1 <= p <= M`.

The entry points use the following data paths, relative to the working directory:

```text
data/MCLP/MCLP_2000_20/MCLP_100_20_valid_Normalization.pkl
data/MCLP_1000_30_normal_Normalization.pkl
```

The first path is used by the training and evaluation routines; the second is used by the rollout baseline. Place your prepared datasets at these paths or update the filename arguments in the corresponding routines.

## Decoding

The model supports `sampling` for stochastic selection and `greedy` for selecting the highest-probability candidate at each step. Configure an instantiated model through its decoding API:

```python
from nets.CacheFormer import set_decode_type

set_decode_type(model, "sampling")
# For greedy inference:
set_decode_type(model, "greedy")
```

Model dimensions, optimizer settings, batch sizes, epochs, and checkpoint paths are configured in `options.py`.

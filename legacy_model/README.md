# Archived CultiNet Neural Model

This directory contains the original `run.py`, `options.py`, and `train.py`, together with the required source files in `nets/`, `problems/`, and `utils/`. These files are byte-for-byte copies of the original workspace sources. File paths and SHA-256 hashes are listed in `source_manifest.json`. The model and algorithms have not been modified.

**The paper's trained weights, original training data, complete historical hyperparameters, and a locked training environment are not included. This archive therefore cannot directly reproduce the paper's CultiNet results.** It is separate from the EH-MCLP baseline experiment entry points in this release.

Training dependencies include `torch`, `numpy`, `Pillow`, `scipy`, `tqdm`, and `tensorboard_logger`. To reuse the archived entry points, supply your own training and validation data and address the following existing interfaces and limitations:

- `run.py` and `train.py` read `data/MCLP/MCLP_2000_20/MCLP_100_20_valid_Normalization.pkl`. The original code uses the same path for training and validation, and the `--val_dataset` option is not applied.
- The rollout baseline reads `data/MCLP_1000_30_normal_Normalization.pkl`. These paths are resolved relative to the working directory used to run the archived entry point.
- The model expects `facilities [B,M,5]`, `users [B,N,2]`, `demand [B,N,1]`, and `p [B]` and `r [B]` with constant values within each batch. The first two facility features are coordinates used for distance calculations. The historical definitions and preprocessing of the remaining three features could not be reliably established. The original random dataset class generates only two facility features and is incompatible with this model as written.
- The implemented decoding modes are `greedy` and `sampling`; `topk` and `topp` appear only in the options. The original training call also passes an incompatible number of arguments to `set_decode_type`, and the `load_model` helper imports the missing `nets.attention_model_old` module. These entry points require further preparation for your data and environment before use.

The objective in `problems/MCLP/problem_MCLP.py` is `0.6 × distance-decayed weighted coverage − 0.4 × service distance`. The distance-decay thresholds are hard-coded as `1000/38686.093359901104` and `2000/38686.093359901104`. This differs from the main release's EH-MCLP objective, `0.35F1 + 0.40F2 + 0.25F3`, and does not implement its candidate eligibility or site-capacity constraints.

This release packages existing source files only. No training, tests, or reproduction runs were performed during packaging, and no newly generated model weights are included.

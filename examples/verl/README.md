# verl Examples

This folder contains an actual Playpen-to-verl integration for GRPO training using [verl](https://github.com/volcengine/verl).

## Install

From the repository root, install Playpen with the optional `verl` dependency:

```bash
pip install -e '.[verl]'
```

If you train local HuggingFace models through `clemcore` backends, also install:

```bash
pip install 'clemcore[huggingface]'
```

## Run the trainer

Set dataset files first (required):

```bash
export PLAYPEN_VERL_TRAIN_FILE=/absolute/path/to/train.parquet
export PLAYPEN_VERL_VAL_FILE=/absolute/path/to/val.parquet
```

Optional overrides:

```bash
export PLAYPEN_VERL_MODEL_PATH=Qwen/Qwen2.5-7B-Instruct
export PLAYPEN_VERL_OUTPUT_DIR=models/verl/qwen2.5-7b
export PLAYPEN_VERL_TOTAL_EPOCHS=3
export PLAYPEN_VERL_TRAIN_BATCH_SIZE=128
```

Then run:

```bash
playpen run examples/verl/grpo_trainer_template.py -l <model-name>
```

Replace `<model-name>` with a model listed in your local `model_registry.json`.

## LoRA training

To train with LoRA instead of full fine-tuning, set `PLAYPEN_VERL_LORA_RANK` to a positive integer:

```bash
export PLAYPEN_VERL_LORA_RANK=16
export PLAYPEN_VERL_LORA_ALPHA=32          # optional, default 16.0
export PLAYPEN_VERL_LORA_TARGET_MODULES=all-linear  # optional, default "all-linear"
```

To resume from an existing LoRA adapter checkpoint:

```bash
export PLAYPEN_VERL_LORA_ADAPTER_PATH=/path/to/adapter
```

Setting `PLAYPEN_VERL_LORA_RANK=0` (the default) disables LoRA and trains the full model.

## Environment variable reference

| Variable | Default | Description |
|---|---|---|
| `PLAYPEN_VERL_TRAIN_FILE` | *(required)* | Path to training data (parquet/jsonl) |
| `PLAYPEN_VERL_VAL_FILE` | *(required)* | Path to validation data (parquet/jsonl) |
| `PLAYPEN_VERL_MODEL_PATH` | auto-detected | HuggingFace model ID or local path |
| `PLAYPEN_VERL_OUTPUT_DIR` | `models/verl/<model>` | Checkpoint output directory |
| `PLAYPEN_VERL_PROJECT` | `playpen-verl` | Project name for logging |
| `PLAYPEN_VERL_EXPERIMENT` | `grpo-<model>` | Experiment name for logging |
| `PLAYPEN_VERL_TRAIN_BATCH_SIZE` | `64` | Total training batch size |
| `PLAYPEN_VERL_MAX_PROMPT_LENGTH` | `512` | Maximum prompt token length |
| `PLAYPEN_VERL_MAX_RESPONSE_LENGTH` | `256` | Maximum response token length |
| `PLAYPEN_VERL_ACTOR_LR` | `1e-6` | Actor learning rate |
| `PLAYPEN_VERL_PPO_MINI_BATCH_SIZE` | `16` | Mini-batch size for PPO updates |
| `PLAYPEN_VERL_PPO_MICRO_BATCH_SIZE_PER_GPU` | `4` | Micro-batch size per GPU |
| `PLAYPEN_VERL_LOGPROB_MICRO_BATCH_SIZE_PER_GPU` | `4` | Log-prob micro-batch size per GPU |
| `PLAYPEN_VERL_CLIP_RATIO_LOW` | `0.2` | PPO clip ratio lower bound |
| `PLAYPEN_VERL_CLIP_RATIO_HIGH` | `0.28` | PPO clip ratio upper bound |
| `PLAYPEN_VERL_ENTROPY_COEFF` | `0.0` | Entropy regularisation coefficient |
| `PLAYPEN_VERL_KL_LOSS_COEF` | `0.001` | KL loss coefficient |
| `PLAYPEN_VERL_ROLLOUT_N` | `4` | Number of rollout samples per prompt |
| `PLAYPEN_VERL_N_GPUS_PER_NODE` | `1` | GPUs per node |
| `PLAYPEN_VERL_NNODES` | `1` | Number of nodes |
| `PLAYPEN_VERL_TOTAL_EPOCHS` | `1` | Training epochs |
| `PLAYPEN_VERL_SAVE_FREQ` | `-1` | Checkpoint save frequency (-1 = end only) |
| `PLAYPEN_VERL_TEST_FREQ` | `1` | Validation frequency (every N epochs) |
| `PLAYPEN_VERL_INFER_BACKEND` | `vllm` | Rollout inference backend |
| `PLAYPEN_VERL_LORA_RANK` | `0` | LoRA rank (0 = full fine-tuning) |
| `PLAYPEN_VERL_LORA_ALPHA` | `16.0` | LoRA alpha scaling factor |
| `PLAYPEN_VERL_LORA_TARGET_MODULES` | `all-linear` | Comma-separated module names or `all-linear` |
| `PLAYPEN_VERL_LORA_ADAPTER_PATH` | *(unset)* | Path to a pre-trained LoRA adapter to resume from |

## What this implementation does

- Validates all configuration before launch, reporting all errors at once with clear messages.
- Confirms that `verl` is installed and prints the detected version.
- Resolves the policy model path from the learner (or from `PLAYPEN_VERL_MODEL_PATH`).
- Builds concrete Hydra overrides for GRPO and launches `python3 -m verl.trainer.main_ppo`.
- Streams verl's stdout/stderr in real time so training progress is visible immediately.
- Supports LoRA training via `PLAYPEN_VERL_LORA_RANK` and related variables.
- Follows the same `BasePlaypenTrainer` interface used by the other examples in this repository.

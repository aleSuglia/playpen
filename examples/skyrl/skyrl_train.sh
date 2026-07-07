set -x

# Colocated GRPO training+generation for Qwen/Qwen3-0.6B on a Playpen environment.
# uv run examples/train/multiply/multiply_dataset.py --output_dir $HOME/data/multiply
# export WANDB_API_KEY=<your_key_here>
# bash examples/train/multiply/run_multiply.sh

export WANDB_API_KEY="null"
export WANDB_MODE="offline"
export RAY_RUNTIME_ENV_HOOK=ray._private.runtime_env.uv_runtime_env_hook.hook

DATA_DIR="/mnt/ceph_rbd/playpen_train_data"
NUM_GPUS=1

uv run -m examples.skyrl.skyrl_playpen \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen3-0.6B" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.inference_engine.num_engines=$NUM_GPUS \
  generator.inference_engine.tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=8 \
  trainer.policy_mini_batch_size=8 \
  trainer.critic_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=8 \
  trainer.micro_train_batch_size_per_gpu=8 \
  trainer.eval_batch_size=8 \
  trainer.eval_before_train=true \
  trainer.eval_interval=5 \
  trainer.ckpt_interval=10 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.inference_engine.backend=vllm \
  generator.inference_engine.run_engines_locally=true \
  generator.inference_engine.weight_sync_backend=nccl \
  generator.batched=false \
  environment.env_class=playpen \
  generator.n_samples_per_prompt=1 \
  generator.inference_engine.gpu_memory_utilization=0.8 \
  trainer.logger="wandb" \
  trainer.project_name="playpen" \
  trainer.run_name="playpen_test" \
  $@

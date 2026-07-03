import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import List, Optional


from playpen import BasePlaypenTrainer


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be an integer, got: {raw!r}")


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Environment variable {key} must be a float, got: {raw!r}")


@dataclass
class VerlGrpoConfig:
    train_file: str
    val_file: str
    model_path: str
    output_dir: str
    project_name: str = "playpen-verl"
    experiment_name: str = "grpo"

    # Core algorithm/data parameters
    train_batch_size: int = 64
    max_prompt_length: int = 512
    max_response_length: int = 256
    actor_lr: float = 1e-6

    # PPO/GRPO optimization parameters
    ppo_mini_batch_size: int = 16
    ppo_micro_batch_size_per_gpu: int = 4
    log_prob_micro_batch_size_per_gpu: int = 4
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.28
    entropy_coeff: float = 0.0
    kl_loss_coef: float = 0.001
    rollout_n: int = 4

    # Runtime/trainer parameters
    n_gpus_per_node: int = 1
    nnodes: int = 1
    total_epochs: int = 1
    save_freq: int = -1
    test_freq: int = 1
    infer_backend: str = "vllm"

    # LoRA parameters (lora_rank=0 disables LoRA)
    lora_rank: int = 0
    lora_alpha: float = 16.0
    lora_target_modules: str = "all-linear"
    lora_adapter_path: str = ""

    @property
    def lora_enabled(self) -> bool:
        return self.lora_rank > 0


_REQUIRED_ENV_VARS = [
    ("PLAYPEN_VERL_TRAIN_FILE", "path to the training parquet/jsonl file"),
    ("PLAYPEN_VERL_VAL_FILE", "path to the validation parquet/jsonl file"),
]


def _validate_config(cfg: VerlGrpoConfig) -> None:
    errors: List[str] = []

    if not cfg.train_file:
        errors.append("PLAYPEN_VERL_TRAIN_FILE is not set — provide the path to the training data file")
    elif not Path(cfg.train_file).exists():
        errors.append(f"PLAYPEN_VERL_TRAIN_FILE path does not exist: {cfg.train_file}")

    if not cfg.val_file:
        errors.append("PLAYPEN_VERL_VAL_FILE is not set — provide the path to the validation data file")
    elif not Path(cfg.val_file).exists():
        errors.append(f"PLAYPEN_VERL_VAL_FILE path does not exist: {cfg.val_file}")

    if not cfg.model_path:
        errors.append(
            "Could not resolve model path. Set PLAYPEN_VERL_MODEL_PATH explicitly, "
            "or ensure your learner model exposes model_id, model_name, or huggingface_id."
        )

    if cfg.train_batch_size <= 0:
        errors.append(f"PLAYPEN_VERL_TRAIN_BATCH_SIZE must be positive, got {cfg.train_batch_size}")

    if cfg.ppo_mini_batch_size > cfg.train_batch_size:
        errors.append(
            f"PLAYPEN_VERL_PPO_MINI_BATCH_SIZE ({cfg.ppo_mini_batch_size}) "
            f"must be <= PLAYPEN_VERL_TRAIN_BATCH_SIZE ({cfg.train_batch_size})"
        )

    if cfg.lora_enabled:
        if cfg.lora_rank <= 0:
            errors.append(f"PLAYPEN_VERL_LORA_RANK must be a positive integer when LoRA is enabled, got {cfg.lora_rank}")
        if cfg.lora_alpha <= 0:
            errors.append(f"PLAYPEN_VERL_LORA_ALPHA must be positive, got {cfg.lora_alpha}")
        if cfg.lora_adapter_path and not Path(cfg.lora_adapter_path).exists():
            errors.append(f"PLAYPEN_VERL_LORA_ADAPTER_PATH does not exist: {cfg.lora_adapter_path}")

    if errors:
        msg = "\n".join(f"  - {e}" for e in errors)
        raise ValueError(f"VerlGrpoTrainer configuration errors:\n{msg}")


class VerlGrpoTrainer(BasePlaypenTrainer):

    def __init__(self, learner):
        super().__init__(learner)

    def learn(self):
        version = metadata.version("verl")
        config = self.build_config()
        _validate_config(config)

        print(f"Detected verl version: {version}")
        print(f"Learner model: {self.learner.name}")
        print(f"Model path for verl: {config.model_path}")
        print(f"Train file: {config.train_file}")
        print(f"Validation file: {config.val_file}")
        print(f"Training output: {config.output_dir}")
        if config.lora_enabled:
            print(f"LoRA enabled — rank={config.lora_rank}, alpha={config.lora_alpha}, target_modules={config.lora_target_modules}")

        self.ensure_parent_dir(config.output_dir)
        cmd = self.build_verl_command(config)

        printable_cmd = " ".join(shlex.quote(c) for c in cmd)
        print("Launching verl with command:")
        print(printable_cmd)

        self._run_streaming(cmd)
        print("verl GRPO training finished.")

    def _run_streaming(self, cmd: List[str]) -> None:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
        finally:
            process.stdout.close()
            return_code = process.wait()

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd)

    def build_config(self) -> VerlGrpoConfig:
        model_path = self.resolve_model_path()

        return VerlGrpoConfig(
            train_file=os.getenv("PLAYPEN_VERL_TRAIN_FILE", ""),
            val_file=os.getenv("PLAYPEN_VERL_VAL_FILE", ""),
            model_path=model_path,
            output_dir=os.getenv("PLAYPEN_VERL_OUTPUT_DIR", f"models/verl/{self.learner.name}"),
            project_name=os.getenv("PLAYPEN_VERL_PROJECT", "playpen-verl"),
            experiment_name=os.getenv("PLAYPEN_VERL_EXPERIMENT", f"grpo-{self.learner.name}"),
            train_batch_size=_env_int("PLAYPEN_VERL_TRAIN_BATCH_SIZE", 64),
            max_prompt_length=_env_int("PLAYPEN_VERL_MAX_PROMPT_LENGTH", 512),
            max_response_length=_env_int("PLAYPEN_VERL_MAX_RESPONSE_LENGTH", 256),
            actor_lr=_env_float("PLAYPEN_VERL_ACTOR_LR", 1e-6),
            ppo_mini_batch_size=_env_int("PLAYPEN_VERL_PPO_MINI_BATCH_SIZE", 16),
            ppo_micro_batch_size_per_gpu=_env_int("PLAYPEN_VERL_PPO_MICRO_BATCH_SIZE_PER_GPU", 4),
            log_prob_micro_batch_size_per_gpu=_env_int("PLAYPEN_VERL_LOGPROB_MICRO_BATCH_SIZE_PER_GPU", 4),
            clip_ratio_low=_env_float("PLAYPEN_VERL_CLIP_RATIO_LOW", 0.2),
            clip_ratio_high=_env_float("PLAYPEN_VERL_CLIP_RATIO_HIGH", 0.28),
            entropy_coeff=_env_float("PLAYPEN_VERL_ENTROPY_COEFF", 0.0),
            kl_loss_coef=_env_float("PLAYPEN_VERL_KL_LOSS_COEF", 0.001),
            rollout_n=_env_int("PLAYPEN_VERL_ROLLOUT_N", 4),
            n_gpus_per_node=_env_int("PLAYPEN_VERL_N_GPUS_PER_NODE", 1),
            nnodes=_env_int("PLAYPEN_VERL_NNODES", 1),
            total_epochs=_env_int("PLAYPEN_VERL_TOTAL_EPOCHS", 1),
            save_freq=_env_int("PLAYPEN_VERL_SAVE_FREQ", -1),
            test_freq=_env_int("PLAYPEN_VERL_TEST_FREQ", 1),
            infer_backend=os.getenv("PLAYPEN_VERL_INFER_BACKEND", "vllm"),
            lora_rank=_env_int("PLAYPEN_VERL_LORA_RANK", 0),
            lora_alpha=_env_float("PLAYPEN_VERL_LORA_ALPHA", 16.0),
            lora_target_modules=os.getenv("PLAYPEN_VERL_LORA_TARGET_MODULES", "all-linear"),
            lora_adapter_path=os.getenv("PLAYPEN_VERL_LORA_ADAPTER_PATH", ""),
        )

    def resolve_model_path(self) -> str:
        explicit = os.getenv("PLAYPEN_VERL_MODEL_PATH")
        if explicit:
            return explicit

        candidates = [
            getattr(self.learner, "model_id", None),
            getattr(self.learner, "model_name", None),
            getattr(self.learner, "name", None),
        ]

        spec = getattr(self.learner, "model_spec", None)
        if spec is not None:
            candidates.extend([
                getattr(spec, "huggingface_id", None),
                getattr(spec, "model_name", None),
            ])

        for value in candidates:
            if isinstance(value, str) and value.strip():
                return value

        return ""

    def build_verl_command(self, cfg: VerlGrpoConfig) -> List[str]:
        cmd = [
            "python3",
            "-m",
            "verl.trainer.main_ppo",
            "algorithm.adv_estimator=grpo",
            "algorithm.use_kl_in_reward=False",
            f"data.train_files={cfg.train_file}",
            f"data.val_files={cfg.val_file}",
            "data.prompt_key=prompt",
            "data.return_raw_chat=True",
            f"data.train_batch_size={cfg.train_batch_size}",
            f"data.max_prompt_length={cfg.max_prompt_length}",
            f"data.max_response_length={cfg.max_response_length}",
            "data.filter_overlong_prompts=True",
            "data.truncation=error",
            f"actor_rollout_ref.model.path={cfg.model_path}",
            "actor_rollout_ref.model.trust_remote_code=True",
            "actor_rollout_ref.model.enable_gradient_checkpointing=True",
            "actor_rollout_ref.actor.strategy=fsdp",
            f"actor_rollout_ref.actor.optim.lr={cfg.actor_lr}",
            f"actor_rollout_ref.actor.ppo_mini_batch_size={cfg.ppo_mini_batch_size}",
            f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={cfg.ppo_micro_batch_size_per_gpu}",
            "actor_rollout_ref.actor.use_kl_loss=True",
            f"actor_rollout_ref.actor.kl_loss_coef={cfg.kl_loss_coef}",
            f"actor_rollout_ref.actor.clip_ratio_low={cfg.clip_ratio_low}",
            f"actor_rollout_ref.actor.clip_ratio_high={cfg.clip_ratio_high}",
            f"actor_rollout_ref.actor.entropy_coeff={cfg.entropy_coeff}",
            f"actor_rollout_ref.rollout.name={cfg.infer_backend}",
            f"actor_rollout_ref.rollout.n={cfg.rollout_n}",
            f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={cfg.log_prob_micro_batch_size_per_gpu}",
            "critic.enable=False",
            "trainer.logger=[console]",
            f"trainer.project_name={cfg.project_name}",
            f"trainer.experiment_name={cfg.experiment_name}",
            f"trainer.default_local_dir={cfg.output_dir}",
            "trainer.val_before_train=False",
            f"trainer.n_gpus_per_node={cfg.n_gpus_per_node}",
            f"trainer.nnodes={cfg.nnodes}",
            f"trainer.save_freq={cfg.save_freq}",
            f"trainer.test_freq={cfg.test_freq}",
            f"trainer.total_epochs={cfg.total_epochs}",
        ]

        if cfg.lora_enabled:
            cmd += [
                f"actor_rollout_ref.model.lora_rank={cfg.lora_rank}",
                f"actor_rollout_ref.model.lora_alpha={cfg.lora_alpha}",
                f"actor_rollout_ref.model.target_modules={cfg.lora_target_modules}",
            ]
            if cfg.lora_adapter_path:
                cmd.append(f"actor_rollout_ref.model.lora_adapter_path={cfg.lora_adapter_path}")

        return cmd

    @staticmethod
    def ensure_parent_dir(path_str: str):
        path = Path(path_str)
        path.mkdir(parents=True, exist_ok=True)


# Backward-compatible alias for users who already used the previous class name.
VerlGrpoTrainerTemplate = VerlGrpoTrainer

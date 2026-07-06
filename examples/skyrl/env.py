from typing import Any, Dict

from skyrl_gym.envs.base_text_env import BaseTextEnv, BaseTextEnvStepOutput


def playpen_observation_to_skyrl(observation: Any) -> Any:
    """Convert a Playpen observation into a SkyRL-compatible observation."""
    if isinstance(observation, dict):
        for key in ("observation", "text", "message", "content"):
            if key in observation:
                return observation[key]

    return observation


class PlaypenEnv(BaseTextEnv):
    """
    Environment for multiplication.
    """

    def __init__(
        self,
        env_config: Dict[str, Any] = {},
        extras: Dict[str, Any] = {},
    ):
        super().__init__()

        assert "reward_spec" in extras, "reward_spec field is required"
        assert "ground_truth" in extras["reward_spec"], (
            "ground_truth is required in reward_spec field"
        )
        self.ground_truth = extras["reward_spec"]["ground_truth"]

        self.max_turns = extras["max_turns"] if "max_turns" in extras else 5

        from clemcore.clemgame import episode_results_folder_callbacks, gym_env

        # Create callbacks to record the interactions in a folder; here we name the folder after the models the agent uses
        callbacks = episode_results_folder_callbacks(
            run_dir="clp-chat",
            result_dir_path="playpen-records",
            player_model_infos="MyAgenticGuesser",
        )

        # By default, player_0 is the learner for single-player games like Wordle
        self._game_env = gym_env(env_config["name"], callbacks=callbacks)

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        new_obs, reward, termination, truncation, info = self._game_env.step(action)

        return BaseTextEnvStepOutput(
            observations=playpen_observation_to_skyrl(new_obs),
            reward=reward,
            done=termination or truncation,
            metadata=info,
        )

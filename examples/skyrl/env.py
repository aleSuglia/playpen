from typing import Any, Dict

from clemcore.backends import load_model
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

        print(env_config)
        self.max_turns = extras["max_turns"] if "max_turns" in extras else 5

        from clemcore.clemgame import episode_results_folder_callbacks, gym_env

        # Create callbacks to record the interactions in a folder; here we name the folder after the models the agent uses
        callbacks = episode_results_folder_callbacks(
            run_dir="clp-chat",
            result_dir_path="playpen-records",
            player_model_infos="MyAgenticGuesser",
        )

        self._game_env = gym_env(
            env_config["name"],
            learner_agent="player_0",
            env_agents={
                "player_1": load_model(
                    "clp-chat", gen_args=dict(temperature=0.7, max_tokens=None)
                )
            },
            callbacks=callbacks,
        )

    def reset(self):
        self.turns = 0
        new_obs, info = self._game_env.reset()
        return playpen_observation_to_skyrl(new_obs), info

    def step(self, action: str) -> BaseTextEnvStepOutput:
        self.turns += 1

        new_obs, reward, termination, truncation, info = self._game_env.step(action)

        return BaseTextEnvStepOutput(
            observations=playpen_observation_to_skyrl(new_obs),
            reward=reward,
            done=termination or truncation,
            metadata=info,
        )


if __name__ == "__main__":
    from clemcore.backends import ModelRegistry

    registry = ModelRegistry.register(
        "clp-chat", backend="openai_compatible", model_id="Qwen/Qwen3.5-4B"
    )
    registry.get_first_model_spec_that_unify_with("clp-chat")
    game_env = PlaypenEnv(env_config=dict(name="taboo"))

    last_obs, info = game_env.reset()
    termination = False
    context_response_pairs: list[tuple] = []
    counter = 0

    while not termination:
        print(f"Round: {counter} ...")
        print(last_obs)
        action = input("Enter your action: ")
        step_output = game_env.step(action)
        context_response_pairs.append((last_obs, action, step_output["reward"]))
        last_obs = step_output["observations"]
        termination = step_output["done"]
        counter += 1

    print(f"Episode took these {len(context_response_pairs)} steps:")
    print("-" * 20)
    for idx, (context, response, reward) in enumerate(context_response_pairs):
        print(f"Step {idx} / Reward {reward:.2f}:")
        print("Describer <- Context:", context)
        print("Describer -> Response:", response)
        print("-" * 20)

    print("Final reward:", reward)

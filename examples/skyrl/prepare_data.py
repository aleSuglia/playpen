import argparse
import os

from datasets import load_dataset


def make_map_fn(split):
    def process_fn(example, idx):
        """Transform each dataset example into the required format by SkyRL.

        Each example should be formatted as:

        data = {
              "data_source": data_source,     # String: Name/identifier of the data source
              "prompt": [                     # List: Conversation format
                  {
                      "role": "user",
                      "content": question,
                  }
              ],
              "env_class": env_class,         # String: Environment class identifier
              "reward_spec": {
                  "method": "rule",           # String: Either "rule" or "reward_model"
                  "ground_truth": solution,   # Expected solution
              },
              "extra_info": {                 # Dict: Optional additional metadata
                  # ... add your own fields here
              },
          }
        """
        game = str(example["game"])
        experiment = str(example["experiment"])
        task_id = int(example["task_id"])

        data_source = f"playpen-data/{split}/{game}/{experiment}"
        # this is strictly handled within Playpen Env by the GameMaster so we don't need to include it in the prompt, but we can include it for clarity.
        question = ""

        # Keep this aligned with the custom registration in examples/skyrl/skyrl_playpen.py
        env_class = "playpen"

        # PlaypenEnv currently requires reward_spec.ground_truth. The game itself computes rewards,
        # so we pass an empty placeholder and preserve instance selectors in extra_info.
        solution = ""

        data = {
            "data_source": data_source,
            "prompt": [
                {
                    "role": "user",
                    "content": question,
                }
            ],
            "env_class": env_class,
            "reward_spec": {
                "method": "rule",
                "ground_truth": solution,
            },
            "extra_info": {
                "dataset": "colab-potsdam/playpen-data",
                "subset": "instances",
                "split": split,
                "index": idx,
                "game": game,
                "experiment": experiment,
                "task_id": task_id,
            },
            # Expose selectors at top-level so they are available as env extras in SkyRL.
            "game": game,
            "experiment": experiment,
            "task_id": task_id,
            "split": split,
            "instance_idx": idx,
        }

        return data

    return process_fn


def main():
    parser = argparse.ArgumentParser(
        description="Convert playpen-data instances to SkyRL parquet files."
    )
    parser.add_argument(
        "--dataset",
        default="colab-potsdam/playpen-data",
        help="Hugging Face dataset name or path.",
    )
    parser.add_argument(
        "--subset",
        default="instances",
        help="Dataset subset/config to load.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for parquet files.",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    dataset_dict = load_dataset(args.dataset, args.subset)

    if "train" not in dataset_dict:
        raise ValueError("Expected a 'train' split in the input dataset.")

    train_dataset = dataset_dict["train"].map(
        function=make_map_fn("train"),
        with_indices=True,
    )
    train_dataset.to_parquet(os.path.join(args.output, "train.parquet"))

    if "validation" in dataset_dict:
        validation_dataset = dataset_dict["validation"].map(
            function=make_map_fn("validation"),
            with_indices=True,
        )
        validation_dataset.to_parquet(os.path.join(args.output, "validation.parquet"))


if __name__ == "__main__":
    main()

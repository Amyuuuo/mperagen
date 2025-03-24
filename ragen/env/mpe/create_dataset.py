import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
import argparse
import datasets

from ragen.env.mpe.env import MultiGoalNavEnv

GIRD_SIZE = 10
NUM_AGENTS = 3
NUM_OBSTACLES = 0
MAX_STEPS = 100


INSTRUCTION_TEMPLATE = """You are agent {agent_id} in a multi-agent navigation system.

Multi-Goal Navigation Task Guide:
Objective: All agents need to collaborate, with each agent reaching a goal point, while avoiding collisions with other agents and obstacles.

Symbol Legend:
. Empty space | # Obstacle | O Goal point | A Yourself | B Other agents
Å You on a goal point | Ḃ Other agent on a goal point

#### Actions
Each agent can perform one of these actions:
- **Stay (0)**: Remain in the current position
- **Left (1)**: Move one cell to the left
- **Right (2)**: Move one cell to the right
- **Down (3)**: Move one cell down
- **Up (4)**: Move one cell up
---

### Reward Mechanism:
- Each move: `{step_penalty}` (penalty applied per step to encourage efficient movement)  
- Collision penalty: `{collision_penalty}` (penalty if an agent tries to occupy the same space as another agent or obstacle)  
- Reaching a goal point: `{goal_reward}` (reward for successfully reaching a goal)  
- All goal points occupied by agents: `{completion_reward}` (bonus reward when all agents reach their goals)  
- Distance penalty per step: `{distance_factor}` (negative reward proportional to the remaining distance to the goal, encouraging shorter paths)  

[Current Observation]:
{observation}

Please decide your next action. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""
templates = {
    'qwen-instruct': '<|im_start|>user\n{prompt}\nAlways output: <think> [Your thoughts] </think> <answer> [your answer] </answer> with no extra text. Strictly follow this format. <|im_end|>\n<|im_start|>assistant\n<think>',
    'base': 'A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks briefly about the reasoning process in the mind and then provides the user with the answer.\nUser: {prompt}\nShow your work in <think> </think> tags. And return the final answer in <answer> </answer> tags, for example <think> [Thoughts] </think> <answer> 1 </answer>\nAssistant: \n<think>'
}


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Generate trajectories using specified environment and policy.")
    parser.add_argument("--env", type=str, default="mpe", help="Environment name (default: 'sokoban').")
    parser.add_argument("--algo", type=str, default="bfs", choices=["bfs"], help="Algorithm to use (default: 'bfs').")
    parser.add_argument("--seed", type=int, default=10000, help="Seed for random number generation (default: 10000).")
    parser.add_argument("--output", type=str, default="data/mpe",
                        help="Output file to save the trajectories (default: 'data/sokoban').")
    parser.add_argument("--train_size", type=int, default=300,
                        help="Number of trajectories to generate (default: 3000).")
    parser.add_argument("--test_size", type=int, default=10, help="Number of trajectories to generate (default: 100).")
    parser.add_argument("--bfs_max_nodes", type=int, default=1000,
                        help="Maximum number of nodes to use for BFS (default: 100000).")  # not using this now. This will usually give the best traj. To compare with SFT, we will try this later.
    parser.add_argument("--prefix", type=str, default='qwen-instruct', choices=['qwen-instruct', 'base'])

    args = parser.parse_args()

    assert args.env == "sokoban", "Unsupported environment: {args.env}"
    assert args.algo == "bfs", "Unsupported algorithm: {args.algo}"
    data_source = args.env

    grid_size, num_agents, num_obstacles, max_steps = int(GIRD_SIZE), int(NUM_AGENTS), int(NUM_OBSTACLES), int(
        MAX_STEPS)
    seeds = range(args.seed, args.seed + args.train_size + args.test_size)
    instructions = []
    for seed in seeds:
        env = MultiGoalNavEnv(
            grid_size=grid_size,
            num_agents=num_agents,
            num_obstacles=num_obstacles,
            max_steps=max_steps
        )
        observation = env.reset(seed=seed, mode='tiny_rgb_array')
        instruction = INSTRUCTION_TEMPLATE.format(
            agent_id=1,
            observation=observation,
            step_penalty=env.STEP_PENALTY,
            collision_penalty=env.COLLISION_PENALTY,
            goal_reward=env.GOAL_REWARD,
            completion_reward=env.COMPLETION_REWARD,
            distance_factor=env.DISTANCE_FACTOR)
        instructions.append(instruction)

    def _create_instance(idx, instruction):
        prompt_formatted = templates[args.prefix].format(prompt=instruction)

        return {
            "data_source": data_source,
            "prompt": [{"role": "user", "content": prompt_formatted}],
            "ability": "bfs",
            "reward_model": {"style": "rule", "ground_truth": {"target": 0, "numbers": [0, 0]}},
            "extra_info": {"split": "train", "index": idx}
        }

    train_dataset = Dataset.from_list(
        [_create_instance(args.seed + i, instructions[i]) for i in range(args.train_size)])
    test_dataset = Dataset.from_list([_create_instance(args.seed + i, instructions[i]) for i in
                                      range(args.train_size, args.train_size + args.test_size)])

    def make_map_fn(split):
        def process_fn(example, idx):
            return example

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)

    train_dataset.to_parquet(os.path.join(args.output, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.output, 'test.parquet'))


if __name__ == "__main__":
    main()

import argparse
import time
import numpy as np
import torch
from model import ACModel#_PSE
import utils


# Parse arguments

parser = argparse.ArgumentParser()

parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=93,
                    help="random seed (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--color", type=str, default="Grey",
                    help="color of env")

args = parser.parse_args()

# Set seed for all randomness sources

utils.seed(args.seed)

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

# Set device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment
# env = utils.make_env(args.env, args.seed)
# for _ in range(args.shift):
#     env.reset()
# print("Environment loaded\n")

# Load agent
model_dir = utils.get_model_dir(args.model)

# Run the agent
print(args.color)
eval_env = utils.make_env(
    "MiniGrid-ColoredKeys"+args.color+"-v0", args.seed + 100000
)

# agent = utils.Agent(eval_env.observation_space, eval_env.action_space, model_dir, args.ipo_model, pse_model = args.pse_model, device=device, argmax = args.argmax)

obs_space, preprocess_obss = utils.get_obss_preprocessor(eval_env.observation_space)

eval_episodes = 100
eval_returns = []
eval_lengths = []

acmodel = ACModel(obs_space, eval_env.action_space).cuda() # states_all, distance是输入数据，不是模型参数，这里是在定义模型
#_PSE
acmodel.load_state_dict(utils.get_model_state(model_dir))
print("Agent loaded\n")
acmodel.eval()

with torch.no_grad():
    for episode in range(eval_episodes):
        obs = eval_env.reset()
        acmodel.reset_hardcode()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            preprocessed_obs = preprocess_obss([obs], device=device)

            dist, _, _ = acmodel(preprocessed_obs, None)

            # 与原 run_comparisons.py 一样，argmax=False
            if args.argmax:
                action = dist.probs.argmax(dim=1).item()
            else:
                action = dist.sample().item()

            obs, reward, done, _ = eval_env.step(action)
            if episode == 0 and steps < 40:
                print(
                    "step:", steps,
                    "action:", action,
                    "probs:", np.round(dist.probs[0].cpu().numpy(), 3),
                    "hardcode:", acmodel.hardcode,
                    "codestep:", acmodel.codestep,
                    "dooropen:", acmodel.dooropen
                )

            total_reward += reward
            steps += 1

        eval_returns.append(total_reward)
        eval_lengths.append(steps)

eval_env.close()

print(
    args.color+" Evaluation | "
    "R:μσmM {:.3f} {:.3f} {:.3f} {:.3f} | "
    "F:μσmM {:.1f} {:.1f} {:.0f} {:.0f} | "
    "Success {:.1%}".format(
        np.mean(eval_returns),
        np.std(eval_returns),
        np.min(eval_returns),
        np.max(eval_returns),
        np.mean(eval_lengths),
        np.std(eval_lengths),
        np.min(eval_lengths),
        np.max(eval_lengths),
        np.mean(np.array(eval_returns) > 0)
    )
)
import inspect

print("Color:", args.color)
print("Argmax:", args.argmax)
print("Seed:", args.seed)
print("Environment:", eval_env.unwrapped.__class__.__name__)
print("Model source:", inspect.getfile(ACModel))
print("Checkpoint:", model_dir)
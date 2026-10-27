import argparse
import time
import torch
import numpy as np
import utils
import copy, math
from model import ACModel_PSE
import torch_ac

env_size = 7

seed = 93
gamma=0.99
# 调用一次collect进行几次psm环境的采集
episodes = 96
num_envs_total = 48 # 这是个超参数,不是几个env,几个env要看reset
p1 = 0.5 # 这是干啥用的?
pse_alpha = 5

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=93,
                        help="seed")
parser.add_argument("--procs", type=int, default=16,
                        help="number of processes (default: 16)")
parser.add_argument("--frames", type=int, default=10**7,
                        help="number of frames of training (default: 1e7)")

parser.add_argument("--epochs", type=int, default=4,
                        help="number of epochs for PPO (default: 4)")
parser.add_argument("--batch-size", type=int, default=256,
                        help="batch size for PPO (default: 256)")
parser.add_argument("--frames-per-proc", type=int, default=None,
                        help="number of frames per process before update (default: 5 for A2C and 128 for PPO)")
parser.add_argument("--discount", type=float, default=0.99,
                    help="discount factor (default: 0.99)")
parser.add_argument("--lr", type=float, default=0.001,
                    help="learning rate (default: 0.001)")
parser.add_argument("--gae-lambda", type=float, default=0.95,
                    help="lambda coefficient in GAE formula (default: 0.95, 1 means no gae)")
parser.add_argument("--entropy-coef", type=float, default=0.01,
                    help="entropy term coefficient (default: 0.01)")
parser.add_argument("--value-loss-coef", type=float, default=0.5,
                    help="value loss term coefficient (default: 0.5)")
parser.add_argument("--max-grad-norm", type=float, default=0.5,
                    help="maximum norm of gradient (default: 0.5)")
parser.add_argument("--optim-eps", type=float, default=1e-8,
                    help="Adam and RMSprop optimizer epsilon (default: 1e-8)")
parser.add_argument("--optim-alpha", type=float, default=0.99,
                    help="RMSprop optimizer alpha (default: 0.99)")
parser.add_argument("--clip-eps", type=float, default=0.2,
                    help="clipping epsilon for PPO (default: 0.2)")
parser.add_argument("--recurrence", type=int, default=1,
                    help="number of time-steps gradient is backpropagated (default: 1). If > 1, a LSTM is added to the model to have memory.")
parser.add_argument("--pse-alpha", type=float, default=2,
                    help="alpha in cme loss for pse")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model to handle text input")

args = parser.parse_args()

args.mem = args.recurrence > 1
utils.seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True)

env_r = utils.make_env("MiniGrid-ColoredKeysRed-v0", args.seed)
env_g = utils.make_env("MiniGrid-ColoredKeysGreen-v0", args.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# teacher PPO-0没事，teacher PPO-1 red 有问题
model_dir = utils.get_model_dir(
    "ppo-modified-s7f600seed10/MiniGrid-ColoredKeysPPO-1"
)

print("seed: ", args.seed)
print(f"Device: {device}\n")
print("Environment loaded\n")
print("training using alpha: ", args.pse_alpha)

agent_r = utils.Agent(
    env_r.observation_space, env_r.action_space,
    model_dir, ipo_model=False, pse_model=False, device=device, argmax=True # ipo=False, argmax=True
)

agent_g = utils.Agent(
    env_g.observation_space, env_g.action_space,
    model_dir, ipo_model=False, pse_model=False, device=device, argmax=True
)

print("Agent loaded\n")

# states1 = np.zeros((env_size,env_size))
# states2 = np.zeros((env_size,env_size))

# Run the agent
def collect_psm(episodes): # 这是收集psm用的环境数量，预设96
    distances = []
    states_r = []
    states_g = []
    for episode in range(episodes):
        # 这是手动reset
        obs_r = env_r.reset()
        obs_g = env_g.reset()
        agent_r.acmodel.reset_hardcode()
        agent_g.acmodel.reset_hardcode()

        done_r = False
        done_g = False

        state_r, actions_r, probs_r = [], [], [] # 实在是不知道存states干啥
        state_g, actions_g, probs_g = [], [], []

        # 拿到pi*
        while not (done_r and done_g):
            if not done_r:
                prob_r, action_r = agent_r.get_probs(obs_r) # env x
                state_r.append(copy.deepcopy(obs_r))
                actions_r.append(int(action_r))
                probs_r.append(prob_r)
                obs_r, reward_r, done_r, _ = env_r.step(action_r)
            if not done_g:
                prob_g, action_g = agent_g.get_probs(obs_g) # env y
                state_g.append(copy.deepcopy(obs_g))
                actions_g.append(int(action_g))
                probs_g.append(prob_g)
                obs_g, reward_g, done_g, _ = env_g.step(action_g)
        # 开始算d
        actions_r = np.array(actions_r)
        actions_g = np.array(actions_g)
        # 这两个暂时用不上（这里用deterministic argmax）
        probs_r = np.array(probs_r)
        probs_g = np.array(probs_g)

        # 计算cv，公式1/2 /Sigma_x|P(x)-Q(x)|，这里deterministic只有一个值，因此就是两个数等不等
        # 原公式 tv = 0.5 * np.abs(probs_r - probs_g).sum()
        tv = np.zeros((len(actions_r), len(actions_g)))
        for x in range(len(actions_r)):
            for j in range(len(actions_g)):
                tv[x, j] = 1-(actions_r[x] == actions_g[j])
            
        # 计算w distance - w distance在这个deterministic的情况下就是下一个状态的距离

        # 迭代计算distance
        n = len(actions_r)
        m = len(actions_g)
        distance = np.zeros((n,m))

        # print("tv = ", tv)
        while True:
            new_distance = np.zeros_like(distance)
            for x in range(n):
                for j in range(m):
                    # 计算d
                    if x==n-1 and j==m-1: # 那如果只有一条走到最后一步呢？
                        new_distance[x, j] = tv[x, j]
                    else:
                        new_distance[x, j] = tv[x, j]+gamma* distance[min(x+1, len(actions_r)-1), min(j+1, len(actions_g)-1)]
            # 判断converge
            if np.max(abs(new_distance-distance))<1e-6:
                break
            distance = new_distance

        # 就是这里red key打印了很多
        if (agent_r.acmodel.codestep != 7
                or agent_g.acmodel.codestep != 7
                or reward_r <= 0
                or reward_g <= 0):
            print(
                "Teacher check:",
                episode,
                "red:", agent_r.acmodel.codestep, reward_r,
                "green:", agent_g.acmodel.codestep, reward_g
            )
            if reward_r <= 0:
                print(
                    "Red failed:",
                    "length:", len(actions_r),
                    "dooropen:", agent_r.acmodel.dooropen,
                    "hardcode:", agent_r.acmodel.hardcode,
                    "last actions:", actions_r[-15:]
                )
            if reward_g<=0:
                print(
                        "Greeb failed:",
                        "length:", len(actions_g),
                        "dooropen:", agent_g.acmodel.dooropen,
                        "hardcode:", agent_g.acmodel.hardcode,
                        "last actions:", actions_g[-15:]
                    )
        distances.append(distance)
        states_r.append(state_r)
        states_g.append(state_g)
    # print(distance, distance.shape)
    return distances, states_r, states_g

# 1, 定义rl用的环境
num_domain1 = math.ceil(p1*num_envs_total) # Number of environments in domain1
num_domain2 = num_envs_total - num_domain1 # Number of environments in domain2

domain1 = 'MiniGrid-ColoredKeysRed-v0'
domain2 = 'MiniGrid-ColoredKeysGreen-v0'

# Environments from domain1
envs1 = []
for i in range(num_domain1):
    envs1.append(utils.make_env(domain1, args.seed + 10000 * i))

# Environments from domain2
envs2 = []
for i in range(num_domain2):
    envs2.append(utils.make_env(domain2, args.seed + 10000 * i))

# All environments
envs = envs1 + envs2
obs_space_r, preprocess_obss_r = utils.get_obss_preprocessor(env_r.observation_space)
# print(preprocess_obss_r)


# 2, 定义model & algorithm (必须在外面定义,不然每次循环都会产生新的algo没法训)
#   初始化模型
acmodel = ACModel_PSE(obs_space_r, env_r.action_space) # states_all, distance是输入数据，不是模型参数，这里是在定义模型
algo = torch_ac.PSEAlgo(envs, acmodel, device, args.frames_per_proc, args.discount, args.lr, args.gae_lambda,
                                    args.entropy_coef, args.value_loss_coef, args.max_grad_norm, args.recurrence,
                                    args.optim_eps, args.clip_eps, args.epochs, args.batch_size, preprocess_obss_r)
# 这好像是继续训练用的?那是不是也不用啊
# try:
#     status = utils.get_status(model_dir)
# except OSError:
#     status = {"num_frames": 0, "update": 0}
# # 继续上一个optimizer?
# if "optimizer_state" in status:
#     algo.optimizer.load_state_dict(status["optimizer_state"])
#     print("Optimizer loaded\n")
status = {"num_frames": 0, "update": 0}

# 3, 训练
num_frames = status["num_frames"]
update = status["update"]
start_time = time.time() # 占位符，参考multi domain train进行修改

# print(acmodel)
# print(algo)
# print(envs)
# print(status)

# onepath不是onepath，是ppo默认训练出来的，一般往下有时往上
# deterministic就是hardcode onepath
save_dir = utils.get_model_dir(
    f"PSE-model0-alpha{args.pse_alpha}-seed{args.seed}"
)#-onepath-deterministic
while num_frames < args.frames:
    update_start_time = time.time()
    # 计算pse
    distances, states_r, states_g = collect_psm(episodes)
    # print("distance = ", distances)
    # print(len(distances), distances[0].shape)
    # print(len(states_r), len(states_r[0]), states_r[0][0])
    # print(len(states_g), len(states_g[0]), states_g[0][0])
    # 得到d之后开始训练
    
    # 收集rl训练数据
    exps, logs1 = algo.collect_experiences()
    # 注意这里pse也训练 **改psealgo update逻辑** - forward训练rl, get_pse训练pse
    logs2 = algo.update_parameters(exps, distances, states_r, states_g, args.pse_alpha)
    logs = {**logs1, **logs2}
    update_end_time = time.time()
    num_frames += logs["num_frames"]
    update += 1
    
    # print logs - 参考multi domain train
    fps = logs["num_frames"] / (update_end_time - update_start_time)
    duration = int(time.time() - start_time)

    rreturn = utils.synthesize(logs["reshaped_return_per_episode"])
    episode_frames = utils.synthesize(logs["num_frames_per_episode"])

    print("U {} | F {:06} | FPS {:04.0f} | D {} | "
        "rR:μσmM {:.2f} {:.2f} {:.2f} {:.2f} | "
        "F:μσmM {:.1f} {:.1f} {:.0f} {:.0f} | "
        "H {:.3f} | V {:.3f} | pL {:.3f} | vL {:.3f} | ∇ {:.3f} | cmeL {:.3f}"
        .format(update, num_frames, fps, duration, rreturn["mean"], rreturn["std"], rreturn["min"], rreturn["max"],
            episode_frames["mean"], episode_frames["std"], episode_frames["min"], episode_frames["max"], logs["entropy"],
            logs["value"], logs["policy_loss"], logs["value_loss"], logs["grad_norm"], logs["pse_loss"])
        )
    # 存权重
    if update % 10 == 0 or num_frames >= args.frames:
        status = {
            "num_frames": num_frames,
            "update": update,
            "model_state": acmodel.state_dict(),
            "optimizer_state": algo.optimizer.state_dict()
        }

        if hasattr(preprocess_obss_r, "vocab"):
            status["vocab"] = preprocess_obss_r.vocab.vocab

        utils.save_status(status, save_dir)
        print("Status saved")

exit()
# evaluation
eval_env = utils.make_env(
    "MiniGrid-ColoredKeysGrey-v0", args.seed + 100000
)

eval_episodes = 100
eval_returns = []
eval_lengths = []

acmodel.eval()

with torch.no_grad():
    for episode in range(eval_episodes):
        obs = eval_env.reset()
        done = False
        total_reward = 0
        steps = 0

        while not done:
            preprocessed_obs = preprocess_obss_r([obs], device=device)

            dist, _, _ = acmodel(preprocessed_obs, None)

            # 与原 run_comparisons.py 一样，argmax=False
            action = dist.sample().item()

            obs, reward, done, _ = eval_env.step(action)

            total_reward += reward
            steps += 1

        eval_returns.append(total_reward)
        eval_lengths.append(steps)

eval_env.close()

print(
    "Grey Evaluation | "
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
print("done")

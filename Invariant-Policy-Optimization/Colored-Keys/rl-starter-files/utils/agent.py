import torch

import utils
from model import ACModel, ACModel_average, ACModel_PSE

import IPython as ipy


class Agent:
    """An agent.

    It is able:
    - to choose an action given an observation,
    - to analyze the feedback (i.e. reward and done state) of its action."""

    def __init__(self, obs_space, action_space, model_dir, ipo_model, device=None, argmax=False, num_envs=1, pse_model = False):
        obs_space, self.preprocess_obss = utils.get_obss_preprocessor(obs_space)
        if ipo_model:
            self.acmodel = ACModel_average(obs_space, action_space)
        elif pse_model:
            self.acmodel = ACModel_PSE(obs_space, action_space)
            # print("this is pse")
        else:    
            self.acmodel = ACModel(obs_space, action_space)
            # print("this is pi*")

        self.device = device
        self.argmax = argmax
        self.num_envs = num_envs

        if self.acmodel.recurrent:
            self.memories = torch.zeros(self.num_envs, self.acmodel.memory_size)

        self.acmodel.load_state_dict(utils.get_model_state(model_dir))
        self.acmodel.to(self.device)
        self.acmodel.eval()
        if hasattr(self.preprocess_obss, "vocab"):
            self.preprocess_obss.vocab.load_vocab(utils.get_vocab(model_dir))

    def get_actions(self, obss):
        preprocessed_obss = self.preprocess_obss(obss, device=self.device)

        # print(type(preprocessed_obss))          # 类型
        # print(preprocessed_obss.keys())         # 有哪些字段，例如 image、text
        # print(len(preprocessed_obss))           # torch_ac 的 DictList：batch 中有多少条 observation
        # print(preprocessed_obss.image.shape)    # 例如 (15, 7, 7, 3)
        # exit()
        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, _, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, _, _ = self.acmodel(preprocessed_obss)
        
        # print(dist) # 这是个categorical,研究一下怎么hardcode
        # print(dist.probs)
        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()

        return actions.cpu().numpy()

    def get_action(self, obs):
            return self.get_actions([obs])[0]

    def get_probs(self, obs, return_value=False):
        preprocessed_obss = self.preprocess_obss([obs], device=self.device)

        with torch.no_grad():
            if self.acmodel.recurrent:
                dist, value, self.memories = self.acmodel(preprocessed_obss, self.memories)
            else:
                dist, value = self.acmodel(preprocessed_obss)
        
        if self.argmax:
            actions = dist.probs.max(1, keepdim=True)[1]
        else:
            actions = dist.sample()
        
        if return_value:
            return dist.probs.cpu().numpy(), actions.cpu().numpy()[0], value
        else:
            return dist.probs.cpu().numpy(), actions.cpu().numpy()[0]
    
    def analyze_feedbacks(self, rewards, dones):
        if self.acmodel.recurrent:
            masks = 1 - torch.tensor(dones, dtype=torch.float).unsqueeze(1)
            self.memories *= masks

    def analyze_feedback(self, reward, done):
        return self.analyze_feedbacks([reward], [done])

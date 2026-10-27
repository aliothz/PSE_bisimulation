import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import torch_ac
import sys
import numpy as np

import IPython as ipy 

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)
# 原版ppo
# class ACModel(nn.Module, torch_ac.RecurrentACModel):
#     def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
#         super().__init__()

#         # Decide which components are enabled
#         self.use_text = use_text
#         self.use_memory = use_memory

#         # Define image embedding
#         self.image_conv = nn.Sequential(
#             nn.Conv2d(3, 16, (2, 2)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU()
#         )
#         n = obs_space["image"][0]
#         m = obs_space["image"][1]
#         self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

#         # Define memory
#         if self.use_memory:
#             self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

#         # Define text embedding
#         if self.use_text:
#             self.word_embedding_size = 32
#             self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
#             self.text_embedding_size = 128
#             self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

#         # Resize image embedding
#         self.embedding_size = self.semi_memory_size
#         if self.use_text:
#             self.embedding_size += self.text_embedding_size

#         # Define actor's model
#         self.actor = nn.Sequential(
#             nn.Linear(self.embedding_size, 64),
#             nn.Tanh(),
#             # nn.ReLU(),
#             nn.Linear(64, action_space.n)
#         )

#         # Define critic's model
#         self.critic = nn.Sequential(
#             nn.Linear(self.embedding_size, 64),
#             nn.Tanh(),
#             # nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#         # Initialize parameters correctly
#         self.apply(init_params)


#     @property
#     def memory_size(self):
#         return 2*self.semi_memory_size

#     @property
#     def semi_memory_size(self):
#         return self.image_embedding_size

#     def forward(self, obs, memory):
#         x = obs.image.transpose(1, 3).transpose(2, 3)
#         x = self.image_conv(x)
#         x = x.reshape(x.shape[0], -1)

#         if self.use_memory:
#             hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
#             hidden = self.memory_rnn(x, hidden)
#             embedding = hidden[0]
#             memory = torch.cat(hidden, dim=1)
#         else:
#             embedding = x

#         if self.use_text:
#             embed_text = self._get_embed_text(obs.text)
#             embedding = torch.cat((embedding, embed_text), dim=1)


#         # Actor
#         x = self.actor(embedding)
#         dist = Categorical(logits=F.log_softmax(x, dim=1))

#         # Critic
#         x = self.critic(embedding)
#         value = x.squeeze(1)

#         return dist, value, memory

#     def _get_embed_text(self, text):
#         _, hidden = self.text_rnn(self.word_embedding(text))
#         return hidden[-1]

#     def reset_hardcode(self):
#         pass
    
# 这是hardcode两条路
class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory
        self.saved_color = None
        self.hardcode = 0
        self.codestep = 0
        self.redroute = [0, 2, 1, 2, 2, 1, 2]
        self.greenroute = [1, 2, 0, 2, 2, 0, 2]
        self.dooropen = False

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)


    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory=None):
        image = obs.image[0]
        # 是否看到了门
        door = image[..., 0] == 4
        
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # Actor
        x = self.actor(embedding)
        logits = F.log_softmax(x, dim=1)
        dist = Categorical(logits=logits)
        # print(logits)

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        front = image[3, 5] # 3*, [类型，颜色，状态]
        # 看到了则命名颜色
        if door.any():
            door = image[door][0] # 1*3->3
            # saved_color=0为红色，=1为绿色
            self.saved_color = int(door[1].item())
            # 门是否开着+面前是否是墙
            if door[-1]==0.:
                self.dooropen = True
        if self.dooropen and (front[0]==2.) and self.hardcode==0:
            # 开始绕行
            self.hardcode = 1
        # self.step +=1
        if self.hardcode ==1:
            if self.saved_color==0 and self.codestep<7:
                logits = torch.zeros((1, 7), device=x.device)
                logits[0, self.redroute[self.codestep]] = 1.
                # print("step: ", self.codestep)
                # print("logits: ", logits)
                self.codestep += 1
            if self.saved_color==1 and self.codestep<7:
                logits = torch.zeros((1, 7), device=x.device)
                logits[0, self.greenroute[self.codestep]] = 1.
                # print("green logits: ", logits)
                self.codestep += 1
            if self.codestep==7:
                self.hardcode = 2
            dist = Categorical(probs=logits)

        # print(logits)
        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def reset_hardcode(self):
        self.saved_color = None
        self.hardcode = 0
        self.codestep = 0
        self.dooropen = False

# 这是hardcode一条路
# class ACModel(nn.Module, torch_ac.RecurrentACModel):
#     def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
#         super().__init__()

#         # Decide which components are enabled
#         self.use_text = use_text
#         self.use_memory = use_memory
#         self.hardcode = 0
#         self.codestep = 0
#         self.route = [0, 2, 1, 2, 2, 1, 2] #[1, 2, 0, 2, 2, 0, 2]
#         self.dooropen = False

#         # Define image embedding
#         self.image_conv = nn.Sequential(
#             nn.Conv2d(3, 16, (2, 2)),
#             nn.ReLU(),
#             nn.MaxPool2d((2, 2)),
#             nn.Conv2d(16, 32, (2, 2)),
#             nn.ReLU(),
#             nn.Conv2d(32, 64, (2, 2)),
#             nn.ReLU()
#         )
#         n = obs_space["image"][0]
#         m = obs_space["image"][1]
#         self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

#         # Define memory
#         if self.use_memory:
#             self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

#         # Define text embedding
#         if self.use_text:
#             self.word_embedding_size = 32
#             self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
#             self.text_embedding_size = 128
#             self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

#         # Resize image embedding
#         self.embedding_size = self.semi_memory_size
#         if self.use_text:
#             self.embedding_size += self.text_embedding_size

#         # Define actor's model
#         self.actor = nn.Sequential(
#             nn.Linear(self.embedding_size, 64),
#             nn.Tanh(),
#             # nn.ReLU(),
#             nn.Linear(64, action_space.n)
#         )

#         # Define critic's model
#         self.critic = nn.Sequential(
#             nn.Linear(self.embedding_size, 64),
#             nn.Tanh(),
#             # nn.ReLU(),
#             nn.Linear(64, 1)
#         )

#         # Initialize parameters correctly
#         self.apply(init_params)


#     @property
#     def memory_size(self):
#         return 2*self.semi_memory_size

#     @property
#     def semi_memory_size(self):
#         return self.image_embedding_size

#     def forward(self, obs, memory=None):
#         image = obs.image[0]
#         # 是否看到了门
#         door = image[..., 0] == 4
        
#         x = obs.image.transpose(1, 3).transpose(2, 3)
#         x = self.image_conv(x)
#         x = x.reshape(x.shape[0], -1)

#         if self.use_memory:
#             hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
#             hidden = self.memory_rnn(x, hidden)
#             embedding = hidden[0]
#             memory = torch.cat(hidden, dim=1)
#         else:
#             embedding = x

#         if self.use_text:
#             embed_text = self._get_embed_text(obs.text)
#             embedding = torch.cat((embedding, embed_text), dim=1)

#         # Actor
#         x = self.actor(embedding)
#         logits = F.log_softmax(x, dim=1)
#         dist = Categorical(logits=logits)
#         # print(logits)

#         # Critic
#         x = self.critic(embedding)
#         value = x.squeeze(1)

#         front = image[3, 5] # 3*, [类型，颜色，状态]
#         # 看到了则命名颜色
#         if door.any():
#             door = image[door][0] # 1*3->3
#             # 门是否开着+面前是否是墙
#             if door[-1]==0.:
#                 self.dooropen = True
#         if self.dooropen and (front[0]==2.) and self.hardcode==0:
#             # 开始绕行
#             self.hardcode = 1
#         # self.step +=1
#         if self.hardcode ==1:
#             if self.codestep<7:
#                 logits = torch.zeros((1, 7), device=x.device)
#                 logits[0, self.route[self.codestep]] = 1.
#                 # print("step: ", self.codestep)
#                 # print("logits: ", logits)
#                 self.codestep += 1
#             if self.codestep==7:
#                 self.hardcode = 2
#             dist = Categorical(probs=logits)

#         # print(logits)
#         return dist, value, memory

#     def _get_embed_text(self, text):
#         _, hidden = self.text_rnn(self.word_embedding(text))
#         return hidden[-1]

#     def reset_hardcode(self):
#         self.hardcode = 0
#         self.codestep = 0
#         self.dooropen = False

class ACModel_PSE(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory

        # Define image embedding
        self.image_conv = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        print(obs_space)
        n = obs_space["image"][0] # 这个size估计pse是一样的，不用改了吧
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        self.actor_pse_shared = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh()
            )
        # Define actor's model
        self.actor = nn.Linear(64, action_space.n)

        self.pse = nn.Sequential(
                    nn.Linear(64, 64), # 接在shared后面
                    # nn.Tanh(),
                    nn.ReLU(),
                    # nn.Linear(64, 1) #输出是embedding，不是数，SimCLR里面的s是embedding similarity,P3公式2
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.pse_output = None

        # Initialize parameters correctly
        self.apply(init_params)


    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # shared embedding
        x = self.actor_pse_shared(embedding)
        # Actor
        actions = self.actor(x)
        dist = Categorical(logits=F.log_softmax(actions, dim=1))

        # PSE 放在get_pse里面，输出只有三个，方便basealgo
        # pse = self.pse(x)
        # self.pse_output = pse

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def get_pse(self, obs, memory=None): # 其实也能写成self.pse_output一个简单getter，但是容易obs对不齐就这么写了
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # shared embedding
        x = self.actor_pse_shared(embedding)

        # PSE
        pse = self.pse(x)

        return pse

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
    



class ACModel_average(nn.Module, torch_ac.RecurrentACModel): 
    def __init__(self, obs_space, action_space, use_memory=False, use_text=False):
        super().__init__()

        # Decide which components are enabled
        self.use_text = use_text
        self.use_memory = use_memory


        # First, define model for first domain
        # Define image embedding
        self.image_conv1 = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64

        # Define memory
        if self.use_memory:
            ValueError("Memory not supported yet")
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            ValueError("Text not supported yet")
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor1 = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic1 = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 1)
        )

        ##############################################################

        # Next, define model for second domain
        # Define image embedding
        self.image_conv2 = nn.Sequential(
            nn.Conv2d(3, 16, (2, 2)),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )

        # Define memory
        if self.use_memory:
            ValueError("Memory not supported yet")
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            ValueError("Text not supported yet")
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        # Define actor's model
        self.actor2 = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic2 = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):

        # Model associated with first domain
        x1 = obs.image.transpose(1, 3).transpose(2, 3)
        x1 = self.image_conv1(x1)
        x1 = x1.reshape(x1.shape[0], -1)

        if self.use_memory:
            ValueError("Memory not supported yet")
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding1 = x1

        if self.use_text:
            ValueError("Text not supported yet")
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # Actor
        x1_actor = self.actor1(embedding1)

        # Critic
        x1 = self.critic1(embedding1)
        value1 = x1.squeeze(1)

        ##############################################################

        # Now model associated with second domain
        x2 = obs.image.transpose(1, 3).transpose(2, 3)
        x2 = self.image_conv2(x2)
        x2 = x2.reshape(x2.shape[0], -1)

        if self.use_memory:
            ValueError("Memory not supported yet")
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding2 = x2

        if self.use_text:
            ValueError("Text not supported yet")
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # Actor
        x2_actor = self.actor2(embedding2)

        # Critic
        x2 = self.critic2(embedding2)
        value2 = x2.squeeze(1)        

        ##############################################################

        # Take average of two models

        # Actor average
        x = (x1_actor+x2_actor)/2 # 训练得是均值
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        # Critic average
        value = (value1+value2)/2

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
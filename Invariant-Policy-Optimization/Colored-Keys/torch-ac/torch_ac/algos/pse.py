import numpy
import torch
import torch.nn.functional as F
import numpy as np
from torch_ac.algos.base import BaseAlgo
import torch.nn.functional as F

pse_dim = 64
# 这个是超参
beta = 3
Lambda = 10
 
def cosine_similarity(a, b):
    a_norm = F.normalize(a, p=2, dim=1)  # 每行除以自己的长度
    b_norm = F.normalize(b, p=2, dim=1)

    similarity = a_norm @ b_norm.T      # (15, 64) @ (64, 15)
                                        # → (15, 15)
    return similarity

def simcls(pos_idx, sim, Tal, Lambda):
    pos_idx = torch.as_tensor(pos_idx, dtype=torch.long, device=sim.device)
    Tal = torch.as_tensor(Tal, dtype=sim.dtype, device=sim.device)
    rows = torch.arange(sim.shape[0], device=sim.device)
    pos_sim = sim[rows, pos_idx]   # (16,) 每行取一个正例
    pos_weight = Tal[rows, pos_idx]  # (16,)

    positive = pos_weight * torch.exp(pos_sim * Lambda)  # (16,)
    # print("pos_idx: ", pos_idx)

    neg_weight = 1-Tal
    negative = neg_weight * torch.exp(sim*Lambda)
    negative[rows, pos_idx] = 0.

    negative = negative.sum(dim=1)
    loss = -torch.log(positive/(positive+negative))
    # print("loss: ", loss)
    return loss.mean()


class PSEAlgo(BaseAlgo):
    """The Proximal Policy Optimization algorithm
    ([Schulman et al., 2015](https://arxiv.org/abs/1707.06347))."""

    def __init__(self, envs, acmodel, device=None, num_frames_per_proc=None, discount=0.99, lr=0.001, gae_lambda=0.95,
                 entropy_coef=0.01, value_loss_coef=0.5, max_grad_norm=0.5, recurrence=4,
                 adam_eps=1e-8, clip_eps=0.2, epochs=4, batch_size=256, preprocess_obss=None,
                 reshape_reward=None):
        # print("adam ", adam_eps, " clip ", clip_eps, " epoch ", epochs, " batch ", batch_size, 
        #       " preprocess ", preprocess_obss, " reshape ", reshape_reward)
        num_frames_per_proc = num_frames_per_proc or 128

        super().__init__(envs, acmodel, device, num_frames_per_proc, discount, lr, gae_lambda, entropy_coef,
                         value_loss_coef, max_grad_norm, recurrence, preprocess_obss, reshape_reward)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size 
        # self.pse_env1 = pse_states_env1
        # self.pse_env2 = pse_states_env2

        assert self.batch_size % self.recurrence == 0

        self.optimizer = torch.optim.Adam(self.acmodel.parameters(), lr, eps=adam_eps)
        self.batch_num = 0
        

    def update_parameters(self, exps, distances, pse_states_env1, pse_states_env2, alpha):
        pse_len = len(distances)
        pse_order = np.random.permutation(pse_len)
        # print('pse sampling order in update: ', pse_order)
        # preprocess pse data
        # pse_env_r = self.preprocess_obss(self.pse_env1, device=self.device)
        # pse_env_g = self.preprocess_obss(self.pse_env2, device=self.device)

        # Collect experiences
        iteration = 0

        for _ in range(self.epochs):
            # Initialize log values

            log_entropies = []
            log_values = []
            log_policy_losses = []
            log_value_losses = []
            log_grad_norms = []
            log_pse_losses = []

            for inds in self._get_batches_starting_indexes():
                # iteration 超出长度的时候跳回第一个
                if iteration>=pse_len:
                    iteration -= pse_len
                # 我觉得得在这做pse？
                batch_pse_states1 = pse_states_env1[pse_order[iteration]]
                batch_pse_states2 = pse_states_env2[pse_order[iteration]]
                distance_batch = distances[pse_order[iteration]]
                episode_len = len(batch_pse_states1)
                batch_pse_embedding1 = []
                batch_pse_embedding2 = []
                # preprocess_obss接收list of dict转成dictlist输入
                preprocessed_obss1 = self.preprocess_obss(batch_pse_states1, device=self.device)
                preprocessed_obss2 = self.preprocess_obss(batch_pse_states2, device=self.device)
                # 计算embedding
                # print("episode len: ", episode_len)
                batch_pse_embedding1 = self.acmodel.get_pse(preprocessed_obss1)
                # print(batch_pse_embedding1.shape)
                batch_pse_embedding2 = self.acmodel.get_pse(preprocessed_obss2)
                # print(batch_pse_embedding2.shape)
                
                # 数据: embedding1 embedding2, states1, states2
                # 计算sim(u, v) cosine similarity
                sim = cosine_similarity(batch_pse_embedding1, batch_pse_embedding2)
                # 用distance计算Tal
                Tal = np.exp(-distance_batch/beta)
                # 根据sim寻找正反样本 16个样本
                # print("distance_batch: ", distance_batch)
                # print("tal: ", Tal)
                pos_idx = distance_batch.argmin(axis=1)
                # 计算exp(lambda*sim) - simcls
                # print('pos_idx: ', pos_idx)
                simcls_loss = simcls(pos_idx, sim, Tal, Lambda)
                # print('loss: ', simcls_loss.item())
                log_pse_losses.append(simcls_loss.item())
                iteration += 1
                # exit()


                # Initialize batch values

                batch_entropy = 0
                batch_value = 0
                batch_policy_loss = 0
                batch_value_loss = 0
                batch_loss = 0

                # Initialize memory

                if self.acmodel.recurrent:
                    memory = exps.memory[inds]

                for i in range(self.recurrence):
                    # Create a sub-batch of experience

                    sb = exps[inds + i]

                    # Compute loss

                    if self.acmodel.recurrent:
                        dist, value, memory = self.acmodel(sb.obs, memory * sb.mask)
                    else:
                        dist, value, _ = self.acmodel(sb.obs)

                    entropy = dist.entropy().mean()

                    ratio = torch.exp(dist.log_prob(sb.action) - sb.log_prob)
                    surr1 = ratio * sb.advantage
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * sb.advantage
                    policy_loss = -torch.min(surr1, surr2).mean()

                    value_clipped = sb.value + torch.clamp(value - sb.value, -self.clip_eps, self.clip_eps)
                    surr1 = (value - sb.returnn).pow(2)
                    surr2 = (value_clipped - sb.returnn).pow(2)
                    value_loss = torch.max(surr1, surr2).mean()

                    loss = policy_loss - self.entropy_coef * entropy + self.value_loss_coef * value_loss

                    # Update batch values

                    batch_entropy += entropy.item()
                    batch_value += value.mean().item()
                    batch_policy_loss += policy_loss.item()
                    batch_value_loss += value_loss.item()
                    batch_loss += loss

                    # Update memories for next epoch

                    if self.acmodel.recurrent and i < self.recurrence - 1:
                        exps.memory[inds + i + 1] = memory.detach()

                # Update batch values

                batch_entropy /= self.recurrence
                batch_value /= self.recurrence
                batch_policy_loss /= self.recurrence
                batch_value_loss /= self.recurrence
                batch_loss /= self.recurrence
                # 这更新方式不行吧，两个forward都不一样
                batch_loss = batch_loss + alpha* simcls_loss

                # Update actor-critic

                self.optimizer.zero_grad()
                batch_loss.backward() # 这里有bug，这个p为什么会是nonetype
                # grad_norm = sum(p.grad.data.norm(2).item() ** 2 for p in self.acmodel.parameters()) ** 0.5
                grad_norm = torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm).item()
                # torch.nn.utils.clip_grad_norm_(self.acmodel.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Update log values

                log_entropies.append(batch_entropy)
                log_values.append(batch_value)
                log_policy_losses.append(batch_policy_loss)
                log_value_losses.append(batch_value_loss)
                log_grad_norms.append(grad_norm)

        # Log some values

        logs = {
            "entropy": numpy.mean(log_entropies),
            "value": numpy.mean(log_values),
            "policy_loss": numpy.mean(log_policy_losses),
            "value_loss": numpy.mean(log_value_losses),
            "grad_norm": numpy.mean(log_grad_norms),
            "pse_loss": numpy.mean(log_pse_losses)
        }

        return logs

    def _get_batches_starting_indexes(self):
        """Gives, for each batch, the indexes of the observations given to
        the model and the experiences used to compute the loss at first.

        First, the indexes are the integers from 0 to `self.num_frames` with a step of
        `self.recurrence`, shifted by `self.recurrence//2` one time in two for having
        more diverse batches. Then, the indexes are splited into the different batches.

        Returns
        -------
        batches_starting_indexes : list of list of int
            the indexes of the experiences to be used at first for each batch
        """

        indexes = numpy.arange(0, self.num_frames, self.recurrence)
        indexes = numpy.random.permutation(indexes)

        # Shift starting indexes by self.recurrence//2 half the time
        if self.batch_num % 2 == 1:
            indexes = indexes[(indexes + self.recurrence) % self.num_frames_per_proc != 0]
            indexes += self.recurrence // 2
        self.batch_num += 1

        num_indexes = self.batch_size // self.recurrence
        batches_starting_indexes = [indexes[i:i+num_indexes] for i in range(0, len(indexes), num_indexes)]

        return batches_starting_indexes

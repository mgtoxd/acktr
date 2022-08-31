import os

import gym
import torch

from Base import BaseChoose
from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.arguments import get_args
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr.storage import RolloutStorage


class CoreChoose(BaseChoose):
    def __init__(self, sizeMap, args):
        super().__init__(sizeMap)
        self.args = args
        self.step = 0
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

        if self.args.cuda and torch.cuda.is_available() and self.args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

        log_dir = os.path.expanduser(self.args.log_dir)
        eval_log_dir = log_dir + "_eval"
        utils.cleanup_log_dir(log_dir)
        utils.cleanup_log_dir(eval_log_dir)

        torch.set_num_threads(1)
        self.device = torch.device("cuda:0" if self.args.cuda else "cpu")

        # envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
        #                      args.gamma, args.log_dir, device, False)
        self.envs = gym.make(self.args.env_name, size=5, Core=self)

        self.actor_critic = Policy(
            self.envs.observation_space.shape,
            self.envs.action_space,
            base_kwargs={'recurrent': self.args.recurrent_policy})
        self.actor_critic.to(self.device)

        if self.args.algo == 'a2c':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic,
                self.args.value_loss_coef,
                self.args.entropy_coef,
                lr=self.args.lr,
                eps=self.args.eps,
                alpha=self.args.alpha,
                max_grad_norm=self.args.max_grad_norm)
        elif self.args.algo == 'acktr':
            self.agent = algo.A2C_ACKTR(
                self.actor_critic, self.args.value_loss_coef, self.args.entropy_coef, acktr=True)

        self.rollouts = RolloutStorage(self.args.num_steps, self.args.num_processes,
                                       self.envs.observation_space.shape, self.envs.action_space,
                                       self.actor_critic.recurrent_hidden_state_size)

        self.obs,self.hash2idx = self.envs.reset(return_info = True)

        self.rollouts.obs[0].copy_(torch.Tensor(self.obs))
        self.rollouts.to(self.device)

        # episode_rewards = deque(maxlen=10)
        #
        # start = time.time()
        # num_updates = int(
        #     self.args.num_env_steps) // self.args.num_steps // self.args.num_processes

    def getHighest(self):
        obs, reward, done, infos = self.envs.step(0)
        # Sample actions
        with torch.no_grad():
            value, action, action_log_prob, recurrent_hidden_states = self.actor_critic.act(
                self.rollouts.obs[self.step], self.rollouts.recurrent_hidden_states[self.step],
                self.rollouts.masks[self.step])

        # Obser reward and next obs


        # if 'episode' in infos.keys():
        #     episode_rewards.append(infos['episode']['r'])

        # If done then clean the history of observations.
        masks = torch.FloatTensor(
            [[0.0] if done else [1.0]])
        bad_masks = torch.FloatTensor(
            [[0.0]])
        self.rollouts.insert(obs, recurrent_hidden_states, action,
                             action_log_prob, value, reward, masks, bad_masks)
        self.step += 1
        if self.step % 5 == 0:
            with torch.no_grad():
                next_value = self.actor_critic.get_value(
                    self.rollouts.obs[-1], self.rollouts.recurrent_hidden_states[-1],
                    self.rollouts.masks[-1]).detach()

            self.rollouts.compute_returns(next_value, self.args.use_gae, self.args.gamma,
                                          self.args.gae_lambda, self.args.use_proper_time_limits)

            value_loss, action_loss, dist_entropy = self.agent.update(self.rollouts)
            self.rollouts.after_update()

        if self.step % 500 == 0:
            save_path = os.path.join(self.args.save_dir, self.args.algo)
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            torch.save([
                self.actor_critic,
                getattr(utils.get_vec_normalize(self.envs), 'obs_rms', None)
            ], os.path.join(save_path, self.args.env_name + ".pt"))

        return self.hash2idx[action]

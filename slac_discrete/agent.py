import os
from collections import deque
from tqdm import tqdm
import numpy as np
import torch
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from slac_discrete.memory import LazyMemory
from slac_discrete.model import LatentNetwork, CateoricalPolicy,\
    TwinnedQNetwork
from slac_discrete.utils import create_feature_actions, calc_kl_divergence,\
    disable_gradients, soft_update, update_params, to_onehot, RunningMeanStats


class SlacDiscreteAgent:
    def __init__(self, env, test_env, log_dir, num_steps=10**7, batch_size=256,
                 latent_batch_size=32, num_sequences=8, lr=0.0003,
                 latent_lr=0.0001, feature_dim=256, latent1_dim=32,
                 latent2_dim=256, hidden_units=(256, 256), memory_size=10**6,
                 gamma=0.99, tau=0.005, target_entropy_ratio=0.98,
                 leaky_slope=0.2, reward_coef=100, start_steps=100000,
                 initial_learning_steps=10000, update_interval=1,
                 log_interval=100, eval_interval=250000, num_eval_steps=125000,
                 max_episode_steps=27000, grad_cliping=None,
                 cuda=True, seed=0):

        self.env = env
        self.test_env = test_env
        self.observation_shape = self.env.observation_space.shape
        self.action_shape = self.env.action_space.shape

        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        self.test_env.seed(2**31-1-seed)
        # torch.backends.cudnn.deterministic = True  # It harms a performance.
        # torch.backends.cudnn.benchmark = False  # It harms a performance.

        self.device = torch.device(
            "cuda" if cuda and torch.cuda.is_available() else "cpu")

        self.latent_net = LatentNetwork(
            self.observation_shape, env.action_space.n, feature_dim,
            latent1_dim, latent2_dim, hidden_units, leaky_slope
            ).to(self.device)

        self.policy_net = CateoricalPolicy(
            num_sequences * feature_dim
            + (num_sequences-1) * env.action_space.n,
            env.action_space.n, hidden_units).to(self.device)

        self.online_q_net = TwinnedQNetwork(
            env.action_space.n, latent1_dim + latent2_dim, hidden_units
            ).to(self.device)

        self.target_q_net = TwinnedQNetwork(
            env.action_space.n, latent1_dim + latent2_dim, hidden_units
            ).to(self.device).eval()

        # Copy parameters of the learning network to the target network.
        self.target_q_net.load_state_dict(self.online_q_net.state_dict())
        # Disable gradient calculations of the target network.
        disable_gradients(self.target_q_net)

        # NOTE: Policy network is updated without the encoder part.
        self.policy_optim = Adam(self.policy_net.parameters(), lr=lr)
        self.q1_optim = Adam(self.online_q_net.Q1.parameters(), lr=lr)
        self.q2_optim = Adam(self.online_q_net.Q2.parameters(), lr=lr)
        self.latent_optim = Adam(self.latent_net.parameters(), lr=latent_lr)

        # Target entropy is -log(1/|A|) * ratio (= maximum entropy * ratio).
        self.target_entropy =\
            -np.log(1.0 / self.env.action_space.n) * target_entropy_ratio
        # We optimize log(alpha), instead of alpha.
        self.log_alpha = torch.zeros(
            1, requires_grad=True, device=self.device)
        self.alpha = self.log_alpha.exp()
        self.alpha_optim = Adam([self.log_alpha], lr=lr, eps=1e-4)

        self.memory = LazyMemory(
            memory_size, num_sequences, self.observation_shape,
            self.device)

        self.log_dir = log_dir
        self.model_dir = os.path.join(log_dir, 'model')
        self.summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(self.summary_dir):
            os.makedirs(self.summary_dir)

        self.writer = SummaryWriter(log_dir=self.summary_dir)
        self.train_return = RunningMeanStats(log_interval)

        self.steps = 0
        self.learning_steps = 0
        self.episodes = 0
        self.best_eval_score = -np.inf
        self.num_actions = env.action_space.n
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.latent_batch_size = latent_batch_size
        self.num_sequences = num_sequences
        self.gamma = gamma
        self.tau = tau
        self.reward_coef = reward_coef
        self.start_steps = start_steps
        self.initial_learning_steps = initial_learning_steps
        self.update_interval = update_interval
        self.log_interval = log_interval
        self.eval_interval = eval_interval
        self.num_eval_steps = num_eval_steps
        self.max_episode_steps = max_episode_steps
        self.grad_cliping = grad_cliping

    def run(self):
        while True:
            self.train_episode()
            if self.steps > self.num_steps:
                break

    def is_update(self):
        return self.steps % self.update_interval == 0\
            and self.steps >= self.start_steps

    def explore(self, state_deque, action_deque):
        # Act with randomness.
        feature_action = self.deque_to_batch(state_deque, action_deque)
        with torch.no_grad():
            action, _, _ = self.policy_net(feature_action, deterministic=False)
        return action.cpu().item()

    def exploit(self, state_deque, action_deque):
        # Act without randomness.
        feature_action = self.deque_to_batch(state_deque, action_deque)
        with torch.no_grad():
            action = self.policy_net(feature_action, deterministic=True)
        return action.cpu().item()

    def reset_deque(self, state):
        # Reset deque filled with the initial state and action 'NOOP'.
        state_deque = deque(maxlen=self.num_sequences)
        action_deque = deque(maxlen=self.num_sequences-1)

        for _ in range(self.num_sequences-1):
            state_deque.append(
                np.zeros(self.observation_shape, dtype=np.uint8))
            action_deque.append(
                np.zeros(self.action_shape, dtype=np.uint8))
        state_deque.append(state)

        return state_deque, action_deque

    def deque_to_batch(self, state_deque, action_deque):
        # Convert deques into batched tensor.
        state = np.array(state_deque, dtype=np.uint8)
        state = torch.ByteTensor(
            state).unsqueeze(0).to(self.device).float() / 255.0

        with torch.no_grad():
            feature = self.latent_net.encoder(state).view(1, -1)

        action = torch.LongTensor(np.array(action_deque)).view(-1, 1)
        onehot_action = to_onehot(
            action, self.num_actions).view(1, -1).to(self.device)
        feature_action = torch.cat([feature, onehot_action], dim=-1)
        return feature_action

    def train_episode(self):
        self.episodes += 1
        episode_return = 0.
        episode_steps = 0

        done = False
        state = self.env.reset()
        self.memory.reset_episode(state)
        state_deque, action_deque = self.reset_deque(state)

        while (not done) and episode_steps <= self.max_episode_steps:

            if self.steps >= self.start_steps:
                action = self.explore(state_deque, action_deque)
            else:
                action = self.env.action_space.sample()

            next_state, reward, done, _ = self.env.step(action)

            clipped_reward = max(min(reward, 1.0), -1.0)
            self.memory.append(action, clipped_reward, next_state, done)

            self.steps += 1
            episode_steps += 1
            episode_return += reward

            if self.is_update():
                # First, learn the latent network only.
                if self.learning_steps < self.initial_learning_steps:
                    with tqdm(range(self.initial_learning_steps)) as pbar:
                        pbar.set_description(
                            "Learning only the latent network")
                        for i in pbar:
                            self.learning_steps += 1
                            self.learn_latent()

                self.learn()

            if self.steps % self.eval_interval == 0:
                self.evaluate()
                self.save_models(os.path.join(self.model_dir, 'final'))

            state_deque.append(next_state)
            action_deque.append(action)

        # We log running mean of training rewards.
        self.train_return.append(episode_return)

        if self.episodes % self.log_interval == 0:
            self.writer.add_scalar(
                'return/train', self.train_return.get(), self.steps)

        print(f'episode: {self.episodes:<4}  '
              f'steps: {episode_steps:<4}  '
              f'return: {episode_return:<5.1f}')

    def learn(self):
        self.learning_steps += 1
        soft_update(self.target_q_net, self.online_q_net, self.tau)

        # Update the latent model.
        self.learn_latent()
        # Update policy and q functions.
        self.learn_sac()

    def learn_latent(self):
        images_seq, actions_seq, rewards_seq, dones_seq =\
            self.memory.sample_latent(self.latent_batch_size)
        latent_loss = self.calc_latent_loss(
            images_seq, actions_seq, rewards_seq, dones_seq)
        update_params(
            self.latent_optim, latent_loss, networks=[self.latent_net],
            retain_graph=False, grad_cliping=self.grad_cliping)

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/latent', latent_loss.detach().item(),
                self.learning_steps)

    def learn_sac(self):
        images_seq, actions_seq, rewards =\
            self.memory.sample_sac(self.batch_size)

        # NOTE: Don't update the encoder part of the policy here.
        with torch.no_grad():
            onehot_actions_seq = to_onehot(actions_seq, self.num_actions)

            # f(1:t+1)
            features_seq = self.latent_net.encoder(images_seq)
            latent_samples, _ = self.latent_net.sample_posterior(
                features_seq, onehot_actions_seq)

            # z(t), z(t+1)
            latents_seq = torch.cat(latent_samples, dim=-1)
            latents = latents_seq[:, -2]
            next_latents = latents_seq[:, -1]
            # a(t)
            actions = onehot_actions_seq[:, -1].long()
            # fa(t)=(x(1:t), a(1:t-1)), fa(t+1)=(x(2:t+1), a(2:t))
            feature_actions, next_feature_actions = create_feature_actions(
                features_seq, actions_seq, self.num_actions)

        q1_loss, q2_loss = self.calc_critic_loss(
            latents, next_latents, actions, next_feature_actions,
            rewards)
        policy_loss, entropies = self.calc_policy_loss(
            latents, feature_actions)
        entropy_loss = self.calc_entropy_loss(entropies.detach())

        update_params(
            self.q1_optim, q1_loss, networks=[self.online_q_net.Q1],
            retain_graph=False, grad_cliping=self.grad_cliping)
        update_params(
            self.q2_optim, q2_loss, networks=[self.online_q_net.Q2],
            retain_graph=False, grad_cliping=self.grad_cliping)
        update_params(
            self.policy_optim, policy_loss, networks=[self.policy_net],
            retain_graph=False, grad_cliping=self.grad_cliping)
        update_params(
            self.alpha_optim, entropy_loss, networks=[],
            retain_graph=False, grad_cliping=None)

        self.alpha = self.log_alpha.exp()

        if self.learning_steps % self.log_interval == 0:
            self.writer.add_scalar(
                'loss/Q1', q1_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/Q2', q2_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/policy', policy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'loss/alpha', entropy_loss.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/alpha', self.alpha.detach().item(),
                self.learning_steps)
            self.writer.add_scalar(
                'stats/entropy', entropies.detach().mean().item(),
                self.learning_steps)

    def calc_latent_loss(self, images_seq, actions_seq, rewards_seq,
                         dones_seq):
        features_seq = self.latent_net.encoder(images_seq)
        onehot_actions_seq = to_onehot(actions_seq, self.num_actions)

        # Sample from posterior dynamics.
        (latent1_post_samples, latent2_post_samples),\
            (latent1_post_dists, latent2_post_dists) =\
            self.latent_net.sample_posterior(features_seq, onehot_actions_seq)
        # Sample from prior dynamics.
        (latent1_pri_samples, latent2_pri_samples),\
            (latent1_pri_dists, latent2_pri_dists) =\
            self.latent_net.sample_prior(onehot_actions_seq)

        # KL divergence loss.
        kld_loss = calc_kl_divergence(latent1_post_dists, latent1_pri_dists)

        # Log likelihood loss of generated observations.
        images_seq_dists = self.latent_net.decoder(
            [latent1_post_samples, latent2_post_samples])
        log_likelihood_loss = images_seq_dists.log_prob(
            images_seq).mean(dim=0).sum()

        # Log likelihood loss of genarated rewards.
        rewards_seq_dists = self.latent_net.reward_predictor([
            latent1_post_samples[:, :-1],
            latent2_post_samples[:, :-1],
            onehot_actions_seq, latent1_post_samples[:, 1:],
            latent2_post_samples[:, 1:]])
        reward_log_likelihoods =\
            rewards_seq_dists.log_prob(rewards_seq) * (1.0 - dones_seq)
        reward_log_likelihood_loss = reward_log_likelihoods.mean(dim=0).sum()

        latent_loss =\
            kld_loss - log_likelihood_loss -\
            self.reward_coef * reward_log_likelihood_loss

        if self.learning_steps % self.log_interval == 0:
            reconst_error = (
                images_seq - images_seq_dists.loc
                ).pow(2).mean(dim=(0, 1)).sum().item()
            reward_reconst_error = ((
                rewards_seq - rewards_seq_dists.loc).pow(2) * (1.0 - dones_seq)
                ).mean(dim=(0, 1)).sum().detach().item()
            self.writer.add_scalar(
                'stats/reconst_error', reconst_error, self.learning_steps)
            self.writer.add_scalar(
                'stats/reward_reconst_error',
                self.reward_coef * reward_reconst_error,
                self.learning_steps)

        if self.learning_steps % (100 * self.log_interval) == 0:
            gt_images = images_seq[0].detach().cpu()
            post_images = images_seq_dists.loc[0].detach().cpu()

            with torch.no_grad():
                pri_images = self.latent_net.decoder(
                        [latent1_pri_samples[:1], latent2_pri_samples[:1]]
                        ).loc[0].detach().cpu()
                cond_pri_samples, _ = self.latent_net.sample_prior(
                    onehot_actions_seq[:1], features_seq[:1, 0])
                cond_pri_images = self.latent_net.decoder(
                        cond_pri_samples).loc[0].detach().cpu()

            images = torch.clamp(torch.cat(
                [gt_images, post_images, cond_pri_images, pri_images],
                dim=-2), 0, 1)

            # Visualize 8 sequences instead of 9 for a better visualization
            # because each row contains 8 images at most in TensorBoard.
            self.writer.add_images(
                'images/gt_posterior_cond-prior_prior',
                images[:8], self.learning_steps)

        return latent_loss

    def calc_critic_loss(self, latents, next_latents, actions,
                         next_feature_actions, rewards):
        # Q(z(t), a(t))
        curr_q1, curr_q2 = self.online_q_net(latents, actions)

        with torch.no_grad():
            _, action_probs, log_action_probs =\
                self.policy_net(next_feature_actions, deterministic=False)

            next_q1, next_q2 = self.target_q_net(next_latents)
            next_q = (action_probs * (
                torch.min(next_q1, next_q2) - self.alpha * log_action_probs
                )).mean(dim=1, keepdim=True)

        # r(t) + gamma * E[Q(z(t+1), a(t+1)) + alpha * H(pi)]
        target_q = rewards + self.gamma * next_q

        # Critic losses are mean squared TD errors.
        q1_loss = 0.5 * torch.mean((curr_q1 - target_q).pow(2))
        q2_loss = 0.5 * torch.mean((curr_q2 - target_q).pow(2))

        if self.learning_steps % self.log_interval == 0:
            mean_q1 = curr_q1.detach().mean().item()
            mean_q2 = curr_q2.detach().mean().item()
            self.writer.add_scalar(
                'stats/mean_Q1', mean_q1, self.learning_steps)
            self.writer.add_scalar(
                'stats/mean_Q2', mean_q2, self.learning_steps)

        return q1_loss, q2_loss

    def calc_policy_loss(self, latents, feature_actions):
        assert not latents.requires_grad and not feature_actions.requires_grad

        # (Log of) probabilities to calculate expectations of Q and entropies.
        _, action_probs, log_action_probs =\
            self.policy_net(feature_actions, deterministic=False)

        # Q for every actions to calculate expectations of Q.
        with torch.no_grad():
            q1, q2 = self.online_q_net(latents, None)

        # Expectations of entropies.
        entropies = -torch.sum(
            action_probs * log_action_probs, dim=1, keepdim=True)
        # Expectations of Q.
        q = torch.sum(torch.min(q1, q2) * action_probs, dim=1, keepdim=True)

        # Policy objective is maximization of (Q + alpha * entropy).
        policy_loss = (- q - self.alpha * entropies).mean()

        return policy_loss, entropies

    def calc_entropy_loss(self, entropies):
        assert not entropies.requires_grad

        # Intuitively, we increse alpha when entropy is less than target
        # entropy, vice versa.
        entropy_loss = -torch.mean(
            self.log_alpha * (self.target_entropy - entropies))
        return entropy_loss

    def evaluate(self):
        num_episodes = 0
        num_steps = 0
        total_return = 0.0

        while True:
            state = self.test_env.reset()

            episode_steps = 0
            episode_return = 0.0
            done = False
            state_deque, action_deque = self.reset_deque(state)

            while (not done) and episode_steps <= self.max_episode_steps:
                action = self.exploit(state_deque, action_deque)

                next_state, reward, done, _ = self.env.step(action)
                num_steps += 1
                episode_steps += 1
                episode_return += reward

                state_deque.append(next_state)
                action_deque.append(action)

            num_episodes += 1
            total_return += episode_return

            if num_steps > self.num_eval_steps:
                break

        mean_return = total_return / num_episodes

        if mean_return > self.best_eval_score:
            self.best_eval_score = mean_return
            self.save_models(os.path.join(self.model_dir, 'best'))

        self.writer.add_scalar(
            'return/test', mean_return, self.steps)
        print('-' * 60)
        print(f'Num steps: {self.steps:<5}  '
              f'return: {mean_return:<5.1f}')
        print('-' * 60)

    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(
            self.latent_net.encoder.state_dict(),
            os.path.join(save_dir, 'encoder.pth'))
        torch.save(
            self.latent_net.state_dict(),
            os.path.join(save_dir, 'latent_net.pth'))
        torch.save(
            self.policy_net.state_dict(),
            os.path.join(save_dir, 'policy_net.pth'))
        torch.save(
            self.online_q_net.state_dict(),
            os.path.join(save_dir, 'online_q_net.pth'))
        torch.save(
            self.target_q_net.state_dict(),
            os.path.join(save_dir, 'target_q_net.pth'))

    def __del__(self):
        self.writer.close()
        self.env.close()

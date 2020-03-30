from collections import deque
import numpy as np
import torch


class LazyFrames:

    def __init__(self, frames):
        self._frames = frames
        self.dtype = frames[0].dtype
        self.out = None

    def _force(self):
        return np.array(self._frames, dtype=self.dtype)

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]


class LazySequenceBuff(dict):
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, num_sequences=8):
        super(LazySequenceBuff, self).__init__()
        self.num_sequences = int(num_sequences)

    def reset(self, state):
        self['state'] = deque([state], maxlen=self.num_sequences+1)
        self['action'] = deque(maxlen=self.num_sequences)
        self['reward'] = deque(maxlen=self.num_sequences)
        self['done'] = deque(maxlen=self.num_sequences)
        self.is_prepared = True

    def append(self, action, reward, next_state, done):
        assert len(self['state']) == len(self['action']) + 1

        self['state'].append(next_state)
        self['action'].append(np.array([action], dtype=np.int64))
        self['reward'].append(np.array([reward], dtype=np.float32))
        self['done'].append(np.array([done], dtype=np.bool))

    def get(self):
        assert len(self['state']) == self.num_sequences + 1

        states = LazyFrames(list(self['state']))
        actions = LazyFrames(list(self['action']))
        rewards = LazyFrames(list(self['reward']))
        dones = LazyFrames(list(self['done']))

        return states, actions, rewards, dones

    def __len__(self):
        return len(self['state'])


class LazyMemory(dict):
    keys = ['state', 'action', 'reward', 'done']

    def __init__(self, capacity, num_sequences, observation_shape, device):
        super(LazyMemory, self).__init__()
        self.capacity = int(capacity)
        self.num_sequences = int(num_sequences)
        self.observation_shape = observation_shape
        self.device = device
        self.reset()

    def reset(self):
        self._p = 0
        self._n = 0
        for key in self.keys:
            self[key] = [None] * self.capacity
        self.buff = LazySequenceBuff(num_sequences=self.num_sequences)

    def reset_episode(self, state):
        self.buff.reset(state)

    def append(self, action, reward, next_state, done):
        assert self.buff.is_prepared, 'Buffer need to be reset before the'\
            'episode starts. Please execute memory.reset_episode()'

        self.buff.append(action, reward, next_state, done)

        if len(self.buff) == self.num_sequences + 1:
            states, actions, rewards, dones = self.buff.get()
            self._append(states, actions, rewards, dones)

        if done:
            # Flag to be sure that buffer is reset before the episode starts.
            self.buff.is_prepared = False

    def _append(self, state, action, reward, done):
        self['state'][self._p] = state
        self['action'][self._p] = action
        self['reward'][self._p] = reward
        self['done'][self._p] = done

        self._n = min(self._n + 1, self.capacity)
        self._p = (self._p + 1) % self.capacity

    def sample_latent(self, batch_size):
        '''
        Returns:
            states_seq  : (N, S+1, *observation_shape) shaped tensor.
            actions_seq : (N, S, 1) shaped tensor.
            rewards_seq : (N, S, 1) shaped tensor.
            dones_seq   : (N, S, 1) shaped tensor.
        '''
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states_seq = np.empty((
            batch_size, self.num_sequences + 1, *self.observation_shape),
            dtype=np.uint8)
        actions_seq = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.int64)
        rewards_seq = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.float32)
        dones_seq = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.bool)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            states_seq[i, ...] = self['state'][index]
            actions_seq[i, ...] = self['action'][index]
            rewards_seq[i, ...] = self['reward'][index]
            dones_seq[i, ...] = self['done'][index]

        states_seq = torch.ByteTensor(states_seq).to(self.device).float()/255.
        actions_seq = torch.LongTensor(actions_seq).to(self.device)
        rewards_seq = torch.FloatTensor(rewards_seq).to(self.device)
        dones_seq = torch.BoolTensor(dones_seq).to(self.device).float()

        return states_seq, actions_seq, rewards_seq, dones_seq

    def sample_sac(self, batch_size):
        '''
        Returns:
            states_seq  : (N, S+1, *observation_shape) shaped tensor.
            actions_seq : (N, S, 1) shaped tensor.
            rewards     : (N, 1) shaped tensor.
        '''
        indices = np.random.randint(low=0, high=self._n, size=batch_size)

        states_seq = np.empty((
            batch_size, self.num_sequences+1, *self.observation_shape),
            dtype=np.uint8)
        actions_seq = np.empty((
            batch_size, self.num_sequences, 1), dtype=np.int64)
        rewards = np.empty((batch_size, 1), dtype=np.float32)

        for i, index in enumerate(indices):
            # Convert LazeFrames into np.ndarray here.
            states_seq[i, ...] = self['state'][index]
            actions_seq[i, ...] = self['action'][index]
            rewards[i, ...] = self['reward'][index][-1]

        states_seq = torch.ByteTensor(states_seq).to(self.device).float()/255.
        actions_seq = torch.LongTensor(actions_seq).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)

        return states_seq, actions_seq, rewards

    def __len__(self):
        return self._n

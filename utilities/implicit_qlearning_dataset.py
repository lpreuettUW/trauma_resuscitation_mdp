import torch


class ImplicitQLearningDataset(torch.utils.data.Dataset):
    def __init__(self, states: torch.FloatTensor, actions: torch.LongTensor, next_states: torch.FloatTensor, rewards: torch.FloatTensor, dones: torch.BoolTensor):
        super().__init__()
        self._states = states
        self._actions = actions
        self._next_states = next_states
        self._rewards = rewards
        self._dones = dones

    def __len__(self) -> int:
        return self._states.size(0)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.LongTensor, torch.FloatTensor, torch.FloatTensor, torch.BoolTensor]:
        # noinspection PyTypeChecker
        return self._states[idx], self._actions[idx], self._next_states[idx], self._rewards[idx], self._dones[idx]

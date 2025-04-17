import torch
from typing import Tuple, Final

from mdp.action import Action
from agents.abstract_agent import AbstractAgent


class NoActionAgent(AbstractAgent):
    def __init__(self, no_action_val: int, num_actions: int):
        super(NoActionAgent, self).__init__(torch.nn.Sequential())
        self._no_action_val: Final[int] = no_action_val
        self._num_actions: Final[int] = num_actions

    # region AbstractAgent

    def get_action(self, env_state: torch.FloatTensor) -> Action | Tuple[Action, ...]:
        """
        Get action.

        :param env_state: Environment state.
        :return: Action.
        """
        actions, _ = self.get_best_action(env_state)
        return actions

    def get_best_action(self, env_state: torch.FloatTensor) -> Tuple[Action, torch.FloatTensor] | Tuple[Tuple[Action, ...], Tuple[torch.FloatTensor, ...]]:
        """
        Get best action.

        :param env_state: Environment state.
        :return: Tuple of Action and Probabilities.
        """
        actions = torch.full((env_state.size(0),), fill_value=self._no_action_val, dtype=torch.long, device=env_state.device)
        probs = torch.zeros((env_state.size(0), self._num_actions), dtype=torch.float, device=env_state.device)
        probs[:, self._no_action_val] = 1.0
        actions = Action(action=actions)
        return actions, probs

    def get_action_probs(self, env_state: torch.FloatTensor) -> torch.FloatTensor | Tuple[torch.FloatTensor, ...]:
        """
        Get action probabilities.

        :param env_state: Environment state.
        :return: Action probabilities.
        """
        _, probs = self.get_best_action(env_state)
        return probs

    # endregion

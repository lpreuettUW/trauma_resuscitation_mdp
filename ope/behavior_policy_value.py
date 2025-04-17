import tqdm
import torch

from utilities.device_manager import DeviceManager
from utilities.ope_trajectory_dataset import OPETrajectoryDataset


class BehaviorPolicyValue:
    @staticmethod
    def ComputeValue(traj_dataset: OPETrajectoryDataset, batch_size: int, gamma: float) -> float:
        """
        Compute the value of the behavior policy.

        :param traj_dataset: The trajectory dataset.
        :param batch_size: The batch size.
        :param gamma: The discount factor.
        :return: The value of the behavior policy.
        """
        device = DeviceManager.get_device()
        traj_dataset.reshape_data(False)
        traj_dataset_data_loader = torch.utils.data.DataLoader(traj_dataset, batch_size=batch_size, shuffle=False)  # Not shuffle has yielded weird results..
        gammas = torch.logspace(0, traj_dataset.num_time_steps - 1, traj_dataset.num_time_steps, base=gamma, device=device).unsqueeze(-1).repeat(batch_size, 1, 1)
        value = 0.0
        for _, _, _, _, rewards, _, _, _, missing_data_mask in tqdm.tqdm(traj_dataset_data_loader, desc='Computing Policy Action Probs',
                                                                         total=len(traj_dataset_data_loader), unit=' batch', position=0, leave=True):
            valid_data_mask = missing_data_mask.logical_not().to(device)
            rewards = rewards.to(device)
            gammas = gammas[:rewards.size(0)] # truncate gammas to match batch size
            # compute discounted rewards
            discounted_rewards = (rewards[valid_data_mask] * gammas[valid_data_mask]).sum()
            value += discounted_rewards.cpu().item()
        value /= len(traj_dataset)
        return value

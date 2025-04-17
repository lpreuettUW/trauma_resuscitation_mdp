import os
import copy
import torch
import mlflow
from tqdm import tqdm
from typing import Literal, Dict
from argparse import ArgumentParser

from agents.implicit_q_learning import ImplicitQLearning
from mdp.trauma_icu_resuscitation.state_spaces.discrete import Components as StateSpaceComponents
from utilities.device_manager import DeviceManager
import utilities.trauma_icu_resuscitation_funcs
from utilities.implicit_qlearning_dataset import ImplicitQLearningDataset


def batch_epoch_evaluation(agent: ImplicitQLearning, dataloader_: torch.utils.data.DataLoader, device_: torch.device, train: bool) -> Dict[str, float]:
    """
    Train the agent on a batch of data.
    :param agent: Implicit Q-Learning agent.
    :param dataloader_: DataLoader containing the batch of data.
    :param device_: Device to use for training.
    :param state_space_ae_: State space autoencoder.
    :param train: Whether to train the agent.
    :return: Losses dictionary.
    """
    losses_dict = None
    agent.train() if train else agent.eval()
    for states, actions, next_states, rewards, dones in dataloader_:
        states, actions, next_states, rewards, dones = states.to(device_), actions.to(device_), next_states.to(device_), rewards.to(device_), dones.to(device_)
        if actions.ndim == 1:
            actions = actions.unsqueeze(-1)
        batch_losses_dict = agent.batch_update(states, actions, rewards, next_states, dones) if train else agent.compute_losses(states, actions, rewards, next_states, dones)
        if losses_dict is None:
            losses_dict = batch_losses_dict
        else:
            for key, value in batch_losses_dict.items():
                losses_dict[key] += value
    # log train losses
    for key in losses_dict.keys():
        losses_dict[key] /= len(dataloader_)
    return losses_dict


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('--gpu', type=str, default='-1', help='specify the GPU to use')
    args = ap.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.autograd.set_detect_anomaly(True)

    device = DeviceManager.get_device()

    # parameters
    # agent
    policy_lr = 1e-4
    critic_lr = 1e-4
    expectile_val_lr = 1e-4
    policy_hidden_size = 128
    critic_hidden_size = 128
    expectile_hidden_size = 128
    agent_weight_decay = 1e-4
    expectile = 0.8
    temperature = 0.1
    clip_norm = 1.0
    tau = 5e-3
    # env
    gamma = 1.0
    num_epochs = 300
    action_type: Literal['binary', 'discrete'] = 'discrete'
    num_actions = 12 if action_type == 'discrete' else 8
    # training
    batch_size = 512
    val_epoch_mod = 10
    num_splits = 10
    initial_weights = None

    # mlflow stuffs
    mlflow_path = os.path.join('file:///', '<path_to_mlruns>', 'mlruns')
    mlflow.set_tracking_uri(mlflow_path)
    experiment_name = f'IQL Trauma ICU: {action_type} Actions'
    run_name = 'Reward Function: sparse - scale = 15.0'
    mlflow_experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = mlflow.create_experiment(experiment_name) if mlflow_experiment is None else mlflow_experiment.experiment_id

    with mlflow.start_run(experiment_id=experiment_id, run_name=run_name) as mlflow_run:
        param_dict = {
            'policy_lr': policy_lr,
            'critic_lr': critic_lr,
            'expectile_val_lr': expectile_val_lr,
            'agent_weight_decay': agent_weight_decay,
            'policy_hidden_size': policy_hidden_size,
            'critic_hidden_size': critic_hidden_size,
            'expectile_hidden_size': expectile_hidden_size,
            'expectile': expectile,
            'temperature': temperature,
            'clip_norm': clip_norm,
            'tau': tau,
            'gamma': gamma,
            'num_epochs': num_epochs,
            'batch_size': batch_size,
            'val_epoch_mod': val_epoch_mod,
            'num_splits': num_splits,
            'reward_fn_name': 'sparse',
        }
        mlflow.log_params(param_dict)
        for split in tqdm(range(num_splits), desc='Split', total=num_splits, unit='split ', colour='green'):
            # load data
            (
                (train_states, train_actions, train_rewards, train_dones, train_next_states, _),
                (val_states, val_actions, val_rewards, val_dones, val_next_states, _),
                (_, _, _, _, _, _)
            ) = utilities.trauma_icu_resuscitation_funcs.load_trauma_icu_resuscitation_data(split, 'iql', action_type)
            # flatten actions for D3QN
            match action_type:
                case 'discrete':
                    train_actions = utilities.trauma_icu_resuscitation_funcs.flatten_discrete_actions(train_actions)
                    val_actions = utilities.trauma_icu_resuscitation_funcs.flatten_discrete_actions(val_actions)
                case 'binary':
                    train_actions = utilities.trauma_icu_resuscitation_funcs.flatten_binary_actions(train_actions)
                    val_actions = utilities.trauma_icu_resuscitation_funcs.flatten_binary_actions(val_actions)
                case _:
                    raise ValueError(f'Invalid action type: {action_type}')
            # create dataloaders
            train_dataset = ImplicitQLearningDataset(train_states.float(), train_actions, train_next_states.float(), train_rewards, train_dones)
            val_dataset = ImplicitQLearningDataset(val_states.float(), val_actions, val_next_states.float(), val_rewards, val_dones)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            # initialize agent with the same initial weights for each split
            iql_agent = ImplicitQLearning(state_dim=len(StateSpaceComponents), action_dim=1,
                                          num_actions=num_actions, policy_hidden_dim=policy_hidden_size,
                                          critic_hidden_dim=critic_hidden_size, expectile_val_hidden_dim=expectile_hidden_size, policy_lr=policy_lr,
                                          critic_lr=critic_lr, expectile_val_lr=expectile_val_lr,
                                          gamma=gamma, expectile=expectile, temperature=temperature, clip_norm=clip_norm, tau=tau,
                                          weight_decay=agent_weight_decay)
            if initial_weights is None:
                initial_weights = iql_agent.get_weights()
            else:
                iql_agent.load_weights(initial_weights)

            best_weights = best_combined_loss = None
            for epoch in tqdm(range(num_epochs), desc='Epoch', total=num_epochs, unit='epoch ', colour='blue'):
                epoch_losses_dict = batch_epoch_evaluation(iql_agent, train_dataloader, device, train=True)
                # update loss dict keys
                epoch_losses_dict = {f'train_{key}_split_{split}': value for key, value in epoch_losses_dict.items()}
                mlflow.log_metrics(epoch_losses_dict, step=epoch)
                # evaluate on validation set
                if (epoch + 1) % val_epoch_mod == 0:
                    val_losses_dict = batch_epoch_evaluation(iql_agent, val_dataloader, device, train=False)
                    # update loss dict keys
                    val_losses_dict = {f'val_{key}_split_{split}': value for key, value in val_losses_dict.items()}
                    mlflow.log_metrics(val_losses_dict, step=epoch)
                    # compute score
                    combined_score = 0
                    for val_loss in val_losses_dict.values():
                        combined_score += 0.8 * val_loss
                    for train_loss in epoch_losses_dict.values():
                        combined_score += 0.2 * train_loss
                    if best_combined_loss is None or combined_score < best_combined_loss:
                        best_combined_loss = val_loss
                        best_weights = iql_agent.get_weights()
            # reload best weights
            iql_agent.load_weights(best_weights)
            # save model
            iql_agent.save_model(f'final_iql_model_split_{split}')

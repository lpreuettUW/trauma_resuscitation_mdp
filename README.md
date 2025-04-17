# Trauma Resuscitation Markov Decision Process (MDP)
This repository accompanies our paper “TBD” and provides all code needed to reproduce the development and analysis of a Markov Decision Process (MDP) for trauma resuscitation. The project models early clinical decision-making for trauma patients using real-world electronic health record (EHR) data and formalizes the problem as a sequential decision process to support reinforcement learning research in critical care.

We provide tools for data processing, MDP construction, and baseline policy evaluation, along with scripts to reproduce the figures and results presented in the paper.

# 📁 Repository Structure
```
trauma_resuscitation_mdp/
│── agents/
│   ├── abstract_agent.py           # Abstract class for all agents
│   ├── abstract_batch_agent.py     # Abstract class for batch learning agents
│   ├── abstract_sequence_agent.py  # Abstract class for sequence agents (NOTE: placeholder)
│   ├── d3qn.py                     # Dueling Double Deep Q-Network agent (D3QN)
│   ├── implicit_q_learning.py      # Implicit Q-Learning agent (IQL)
│   ├── no_action_agent.py          # Agent that takes no action
│   ├── random_action_agent.py      # Agent that takes random actions
│── datasets/
│   ├── trauma_icu_resuscitation/
│   │   ├── extract_trajectories.py                 # Clean, Preprocess, and Extract Trajectories
│   │   ├── _data_manager.py/                       # Loads Trauma ICU Resuscitation data
│   │   ├── _trajectory_extraction_utilities.py     # Utilities for extracting trajectories
│   │   ├── stratified_splits/                      # Stratified splits for train/val/test
│── mdp/
│   │── policies/
│   │   ├── dueling_dqn.py                  # Policy for D3QN agent
│   │   ├── next_best_action_policy.py      # Policy for IQL agent
│   │── trauma_icu_resuscitation/
│   │   ├── action_spaces/
│   │   │   ├── binary.py                   # Binary action space
│   │   │   ├── discrete.py                 # Discrete action space (NOTE: this is the final action space)
│   │   ├── state_spaces/
│   │   │   ├── discrete.py                 # Discrete state space
│   │── action.py
│── notebooks/
│   ├── cohort_eda.ipynb                    # Exploratory Data Analysis (EDA) of the cohort data
│   ├── create_splits.ipynb                 # Create stratified splits for train/val/test
│   ├── extracted_trajectory_eda.ipynb      # EDA of the extracted trajectories
│   ├── interventions_eda.ipynb             # EDA of the interventions
│   ├── ivf_returns.ipynb                   # Analyze returns for MDP on our dataset
│   ├── ope_inspection.ipynb                # Analyze the OPE results
│   ├── policy_inspection.ipynb             # Analyze the behavior and learned policies
│   ├── vitals_and_labs_eda.ipynb           # EDA of the vitals and labs
│── ope/
│   ├── abstract_ope_method.py              # Abstract class for OPE methods
│   ├── behavior_policy_value.py            # Compute behavior policy value estimate
│   ├── fqe.py                              # Fitted Q-Evaluation (FQE) method
│   ├── magic.py                            # Model and Guided Importance sampling Combining (MAGIC) method
│── utilities/
│   ├── device_manager.py                   # Torch device manager for GPU/CPU usage
│   ├── implicit_qlearning_dataset.py       # Dataset for D3QN and IQL
│   ├── ope_trajectory_dataset.py           # Dataset for OPE
│   ├── sequence_agent_interface.py         # Placeholder
│   ├── trauma_icu_resuscitation_funcs.py   # Utilities for the Trauma ICU Resuscitation dataset
│── do_ope.py       # Run OPE
│── train_d3qn.py   # Train D3QN agent
│── train_iql.py    # Train IQL agent
│── README.md
│── requirements.txt
│── LICENSE
```
# 🏗️ Installation (Using Conda)
## 1️⃣ Clone the repository
```
git clone https://github.com/lpreuettUW/trauma_resuscitation_mdp.git
cd trauma_resuscitation_mdp
```
## 2️⃣ Create and activate a conda environment
Ensure you have Codna installed, then create an environment with Python 3.10:
```
conda create -n trauma_resuscitation_mdp python=3.10
conda activate trauma_resuscitation_mdp
```
## 3️⃣ Install dependencies
```
pip install -r requirements.txt
```
## 4️⃣ Download the Trauma ICU Resuscitation dataset
Download the dataset from TBD.
## 5️⃣ Update Paths
Replace all instances of <path_to_dataset> in the codebase with the path to your downloaded dataset.<br>
Replace all instances of <path_to_repo> with the path to your cloned repository.<br>
Finally, replace all instances of <path_to_mlruns> with your desired directory for storing MLflow experiments.
## (Optional) 6️⃣ Create splits
If you want to create your own stratified splits for train/val/test, remove the contents of `datasets/trauma_icu_resuscitation/stratified_splits` before running the `create_splits.ipynb` notebook. 
# 🚀 Running Experiments
## 1️⃣ Train D3QN
To train the D3QN agent, run the following command:
```
python train_d3qn.py
```
## 2️⃣ Train IQL
To train the IQL agent, run the following command:
```
python train_iql.py
```
## 3️⃣ Run OPE
To evaluate a given agent using MAGIC, run the following command:
```
python do_ope.py --run_type magic --action_type discrete --agent_type <agent_type> --agent_runid <mlflow_run_id_of_agent>
```

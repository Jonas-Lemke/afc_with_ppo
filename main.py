"""
This file contains the main entry point of the project where parameters can be set.
"""


import datetime
import torch
from torch import optim
# from torch.utils.tensorboard import SummaryWriter

from ppo.network import ActorCriticNetwork
from ppo.environment import WindTunnelEnv
from ppo.training import PPOTraining

from utils import mkdir
from utils import create_file
from utils import write_file

# from tests.test_environments.test_env_01 import TestEnv01  # environment to test ppo implementation
# from tests.test_environments.test_env_02 import TestEnv02  # environment to test ppo implementation
# from tests.test_environments.test_env_03 import TestEnv03  # environment to test ppo implementation
# from tests.test_environments.test_env_04 import TestEnv04  # environment to test ppo implementation

### Parameter Configuration ###

HIDDEN_SIZE         = 64        # Number of neurons in hidden layer in NN
LEARNING_RATE       = 1e-3      # Learning rate for adam optimizer
GAMMA               = 0.99      # Discount factor to calculate returns
GAE_LAMBDA          = 0.95      # Smoothing factor for GAE (high = more accuracy and dependence on future rewards; lower = less variance but higher bias and dependence on immediate rewards)
EPSILON_CLIP        = 0.2       # Used to clip the ratio between new and old policy
CRITIC_DISCOUNT     = 0.01      # Used to scale down critic loss (critic loss is bigger than the actor loss and needs to be scaled down)
ENTROPY_BETA        = 0.0005      # Amount of importance given to the entropy bonus (low = stronger exploitation; high = stronger exploration)
PPO_STEPS           = 128        # Number of transitions sampled for each training iteration = batch_size (should be multiple of MINI_BATCH_SIZE)
MINI_BATCH_SIZE     = 4         # Number of samples which are randomly selected from total amount of stored data
PPO_UPDATE_EPOCHS   = 4         # Number of pass over entire batch of training data
PPO_TRAIN_EPOCHS    = 5000    # Limit epochs for the PPO training loop
SAVE_INTERVAL       = 50        # interval for model checkpoints

DEBUG               = True      # If True extra statistics are written during ppo update
LOAD_MODEL          = False     # Select to load an existing torch model

DEBUG_FILE          = "extra_update_data.txt"
PARAMS_FILE         = "parameter.txt"
BATCH_FILE          = "batch_data.txt"
UPDATE_FILE         = "update_data.txt"

MODEL_LOAD_PATH     = "./logs/run_2025_02_14_13_25_52/model_checkpoints/epoch_2000_torch_model"

if __name__ == "__main__":
    
    ### NI USB-6281 I/O channel names ###
    names_output_channel = ["Dev1/ao0", "Dev1/ao1"]  # ao0: reservoir; ao1: valves
    num_actions = 1

    # names_input_channel = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3",
    #                        "Dev1/ai4", "Dev1/ai5", "Dev1/ai6",  "Dev1/ai7", "Dev1/ai8"]  # ai0: volume flow; ai1 - ai8: tau
    # num_states = 8

    names_input_channel = ["Dev1/ai0", "Dev1/ai1", "Dev1/ai2", "Dev1/ai3",
                           "Dev1/ai4", "Dev1/ai5", "Dev1/ai6",  "Dev1/ai7"]  # ai0: volume flow; ai1 - ai7: tau
    num_states = 7

    ### Create log folder ###
    date = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    models_dir = mkdir('.', f'logs/run_{date}/model_checkpoints')
    log_dir = mkdir('.', f'logs/run_{date}')
    
    # Create log file for used Parameters
    fname_params = f'{log_dir}/{PARAMS_FILE}'
    create_file(fname_params, ['HIDDEN_SIZE', 'LEARNING_RATE', 'GAMMA', 
                               'GAE_LAMBDA', 'PPO_EPSILON_CLIP', 
                               'CRITIC_DISCOUNT', 'ENTROPY_BETA', 'PPO_STEPS', 
                               'MINI_BATCH_SIZE', 'PPO_UPDATE_EPOCHS', 
                               'PPO_TRAIN_EPOCHS'])
    
    # Create log file for PPO Batch Data
    fname_batch = f'{log_dir}/{BATCH_FILE}'
    action_names = ["Action"]
    state_names = names_input_channel[1:]  # List with names of mems sensor channels
    tau_state_names = [f'tau_{sn}' for sn in state_names]
    gamma_state_names = [ f'epoch_gamma_{sn}' for sn in state_names]
    create_file(fname_batch, ["frame_idx", "train_epoch", "step", 
                              "epoch_total_reward", "gae_state_return", 
                              "mc_state_return", "state_value", "reward", 
                              "normalized_gae_advantage", 
                              "normalized_mc_advantage", "act_log_prob", 
                              "dist_prob"] + action_names + state_names + ["volume_flow"] + tau_state_names +
                ["epoch_mean_gamma"] + gamma_state_names)
    
    # Create log file for PPO Update Data
    fname_update = f'{log_dir}/{UPDATE_FILE}'
    create_file(fname_update, ["frame_idx", "train_epoch", "sum_loss_actor", 
                               "sum_loss_critic", "sum_loss_total", 
                               "sum_entropy"])
    
    # Create log file for PPO Update extra Data (debugging info)
    if DEBUG:
        fname_update_extra = f'{log_dir}/{DEBUG_FILE}'
        create_file(fname_update_extra, 
                    ["frame_idx", "train_epoch", "update_epoch", "mini_batch", 
                     "update_mean_returns", "update_mean_advantages", 
                     "update_losses_actor", "update_losses_critic", 
                     "update_losses_total", "update_entropies"])
    else:
        fname_update_extra = None
    
    ### Write used Parameters to file ###
    write_file(f'{log_dir}/{PARAMS_FILE}', 
               [f'{HIDDEN_SIZE}', f'{LEARNING_RATE}', f'{GAMMA}', 
                f'{GAE_LAMBDA}', f'{EPSILON_CLIP}', f'{CRITIC_DISCOUNT}', 
                f'{ENTROPY_BETA}', f'{PPO_STEPS}', f'{MINI_BATCH_SIZE}', 
                f'{PPO_UPDATE_EPOCHS}', f'{PPO_TRAIN_EPOCHS}'])
    
    ### Autodetect CUDA ###
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f'\nDevice: {device}\n')
    
    ### Prepare environment ###
    env = WindTunnelEnv(num_states=num_states, num_actions=num_actions,
                        output_channel=names_output_channel,
                        input_channel=names_input_channel)
    # env = TestEnv03(num_states=num_states, num_actions=num_actions)  # TEST

    ### Prepare Network ###
    model = ActorCriticNetwork(num_inputs=num_states, num_outputs=num_actions, hidden_size=HIDDEN_SIZE).to(device)
    print(model)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    ### Load model ###
    if LOAD_MODEL:
        model.load_state_dict(torch.load(MODEL_LOAD_PATH))
        print("\n##### Model loaded #####\n")
    
    policy_trainer = PPOTraining(env=env, device=device, model=model, 
                                 optimizer=optimizer, 
                                 ppo_train_epochs=PPO_TRAIN_EPOCHS, ppo_steps=PPO_STEPS, 
                                 ppo_update_epochs=PPO_UPDATE_EPOCHS, gamma=GAMMA, 
                                 gae_lambda=GAE_LAMBDA, epsilon_clip=EPSILON_CLIP, 
                                 mini_batch_size=MINI_BATCH_SIZE, entropy_beta=ENTROPY_BETA,
                                 critic_discount=CRITIC_DISCOUNT, save_interval=SAVE_INTERVAL,
                                 chkpnt_dir=models_dir, fname_batch=fname_batch, 
                                 fname_update=fname_update, 
                                 fname_update_extra=fname_update_extra, debug = DEBUG)
    
    ### Start PPO training ###
    
    print("\n\n##### Start Of Training #####\n")
    
    policy_trainer.train()
    
    print("\n##### End Of Training #####\n\n")
    
    ### Stop environment after training ###
    env.end_measurement()

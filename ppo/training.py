"""
This file contains PPO training algorithm
"""

import time
import torch
import numpy as np

from utils import write_file


class PPOTraining:
    def __init__(self, env, device, model, optimizer, ppo_train_epochs, ppo_steps, 
                 ppo_update_epochs, gamma, gae_lambda, epsilon_clip, 
                 mini_batch_size, entropy_beta, critic_discount, 
                 save_interval, chkpnt_dir, fname_batch, fname_update, 
                 fname_update_extra, debug):
        """
        Initializes PPOTraining object.
        """
        self.env = env
        self.device = device
        self.model = model
        self.optimizer = optimizer
        
        self.ppo_train_epochs = ppo_train_epochs
        self.ppo_steps = ppo_steps
        self.ppo_update_epochs = ppo_update_epochs
        
        self.mask = torch.cat((torch.ones((self.ppo_steps - 1, 1), 
                                          dtype=torch.float32, 
                                          device=self.device), 
                               torch.zeros((1, 1), dtype=torch.float32, 
                                           device=self.device)))
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        
        self.epsilon_clip = epsilon_clip
        self.mini_batch_size = mini_batch_size
        self.entropy_beta = entropy_beta
        self.critic_discount = critic_discount
        
        self.chkpnt_dir = chkpnt_dir
        self.save_interval = save_interval
        
        self.fname_batch = fname_batch
        self.fname_update = fname_update
        self.fname_update_extra = fname_update_extra
        
        self.debug = debug
    
    @staticmethod
    def normalize(x):
        """
        Used to normalize the advantage to 0.
        (Theoretically not necessary but decreases variance of advantages 
        and makes convergence more stable and faster in practice)
        """
        x -= x.mean()
        x /= (x.std() + 1e-8)
        return x

    def compute_gae_returns(self, next_value, rewards, values):
        """
        Calculates GAE returns (return = GAE advantage + old estimated state value)
        
        GAE Algorithm
        
        1)  mask = 0 if: state = terminal (episode over)
            mask = 1 else
        2)  initialize gae = 0 
            and loop backward from last step in data
        3)  set delta
            delta = reward + gamma * next_state_value * mask - state_value
        4)  update gae
            gae = delta + gammma * lambda * mask * gae
        5)  return for state and action
            return(s,a) = gae + state_value
        6)  reverse the list to returns back to the correct order
        """
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * self.mask[step] - values[step]
            gae = delta + self.gamma * self.gae_lambda * self.mask[step] * gae
            # prepend to get correct order back
            returns.insert(0, gae + values[step])
        return returns

    def compute_mc_returns(self, rewards):
        """
        Calculates the return of each state using Monte Carlo 
        (easier to understand but not as good as GAE)
        -> just used for debugging and written statistics
        """
        discounted_reward = 0
        returns = []
        for step in reversed(range(len(rewards))):
            discounted_reward = rewards[step] + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)
        return returns

    def ppo_iter(self, states, actions, act_log_probs, returns, advantage):
        """
        Helper function that divides the collected batches
        into smaller mini-batches for optimization.
        """
        
        # function takes alot of time -> see if it can be optimized
        
        batch_size = states.size(0)
        # generates random mini-batches until full batch is covered
        for _ in range(batch_size // self.mini_batch_size):
            rand_ids = np.random.randint(0, batch_size, self.mini_batch_size)
            yield states[rand_ids, :], actions[rand_ids, :], act_log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
    
    def ppo_update(self, frame_idx, train_epoch, states, actions, act_log_probs, returns, advantages):
        """
        Function to update the policy and value network.
            1. sample enough random mini-batches to cover all batch data
            2. pass state into network, obtain action, value, entropy and new_log_probs
            3. calculate surrogate policy loss and mean squared error value loss
            4. backpropagate the total loss through the network using SGD
        """
        sum_loss_actor = 0.0
        sum_loss_critic = 0.0
        sum_loss_total = 0.0
        sum_entropy = 0.0
    
        for update_epoch in range(1, self.ppo_update_epochs + 1):
            
            # print(f'\n##### Update Epoch: {update_epoch} #####\n')
                    
            mb_step = 1  # mini-batch step
            
            # grabs random mini-batches several times until we have covered all data
            for mb_states, mb_actions, mb_old_log_probs, mb_returns, mb_advantages in self.ppo_iter(states, actions, act_log_probs, returns, advantages):
                
                # model prediction of dist and value for all states in mini-batch
                mb_dists, mb_values = self.model(mb_states)
                
                entropy = mb_dists.entropy().mean()
                
                mb_new_log_probs = mb_dists.log_prob(mb_actions)
                
                ### Calculate surrogate losses ###
                ratio = (mb_new_log_probs - mb_old_log_probs).exp()
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon_clip, 1.0 + self.epsilon_clip) * mb_advantages
    
                ### Calculate actor and critic losses ###
                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (mb_returns - mb_values).pow(2).mean()
                
                ### Calculate objective function ###
                loss = self.critic_discount * critic_loss + actor_loss - self.entropy_beta * entropy

                ### Perform back propagation ### ~0.004
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                ### Track statistics ###
                sum_loss_actor += actor_loss
                sum_loss_critic += critic_loss
                sum_loss_total += loss
                sum_entropy += entropy
                
                # Write extra statistics if DEBUG is True (debugging info)
                if self.debug:
                    write_file(self.fname_update_extra, 
                               [frame_idx, train_epoch, update_epoch, mb_step, 
                                mb_returns.mean().item(), 
                                mb_advantages.mean().item(), actor_loss.item(),
                                critic_loss.item(), loss.item(), 
                                entropy.item()])
                
                mb_step += 1
        
        num_updates = self.ppo_update_epochs * (self.ppo_steps/self.mini_batch_size)
        
        ppo_update_data = [frame_idx, train_epoch, 
                           sum_loss_actor.item()/num_updates, 
                           sum_loss_critic.item()/num_updates, 
                           sum_loss_total.item()/num_updates, 
                           sum_entropy.item()/num_updates]
    
        return ppo_update_data

    def collect_batch(self):
        """
        Collects a batch of PPO_STEPS (should be multiple of MINI_BATCH_SIZE) 
        by acting in the environment with current policy
        """
        ### Reset the environment ###
        # state = self.env.reset()
        state, vol_flow, tau = self.env.reset()
        
        ### Batch data ###
        act_log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        dist_probs = []
        vol_flows = []
        taus = []

        for step in range(self.ppo_steps):

            step_start_time = time.perf_counter()  # for fixed time for step
    
            ### Get action and value prediction ### 0.0008 s (virtual env.)
            state = torch.FloatTensor(state).to(self.device)
            dist, value = self.model(state)
            action = dist.sample()

            # print(f'action: {action}')  # TEST
    
            ### Pass action to environment and obtain reward and next state ### ~ 0.0002 s (virtual env.)
            # next_state, reward, _ = self.env.step(bool(action.item()))
            next_state, reward, _, next_vol_flow, next_tau = self.env.step(bool(action.item()))


            ### Get action-log-distribution ### ~ 0.0004 s (virtual env.)
            act_log_prob = dist.log_prob(action)
            
            ### Store data ### ~ 0.0001 s (virtual env.)
            act_log_probs.append(act_log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(self.device))
            states.append(state)
            actions.append(action)
            dist_probs.append(dist.probs)  # (debugging info)
            vol_flows.append(vol_flow)
            taus.append(tau)

            vol_flow = next_vol_flow
            tau = next_tau

            state = next_state

            ### Ensure step takes at least 0.005 s ###
            while(time.perf_counter() - step_start_time) < 0.005:
                pass

        return act_log_probs, values, states, actions, rewards, dist_probs, next_state, vol_flows, taus
    
    
    def calc_batch_gamma(self, taus):
        """
        Calculates the forward flow fraction of a batch for each sensor
        -> used for written statistics
        """
        num_sensors = len(taus[0])
        gamma = []  # list with gamma of each sensor
        for idx in range(num_sensors):
            sensor_values = [sublist[idx] for sublist in taus]
            pos_count = sum(1 for value in sensor_values if value > 0)
            sensor_gamma = pos_count/self.ppo_steps
            gamma.append(sensor_gamma)
            
        return gamma
        
    def train(self):
        """
        Main PPO training loop 
        with batch colletion and network update for PPO_TRAIN_EPOCHS epochs
        """
        frame_idx = 0

        ### Training ###

        for train_epoch in range(1, self.ppo_train_epochs + 1):
            
            print(f'\n##### Training Epoch: {train_epoch} #####\n')

            ### Collect new batch ###
            act_log_probs, values, states, actions, rewards, dist_probs, next_state, vol_flows, taus = self.collect_batch()

            ### Calculate returns for the batch using GAE ###
            next_state = torch.FloatTensor(next_state).to(self.device) # convert state to torch tensor
            _, next_value = self.model(next_state)  # get model value prediction

            gae_returns = self.compute_gae_returns(next_value, rewards, values)
            mc_returns = self.compute_mc_returns(rewards)  # monte carlo returns (just for debugging)

            ### Detach and concatenat tensor lists ###
            rewards = torch.cat(rewards).detach()
            act_log_probs = torch.cat(act_log_probs).detach()
            values = torch.cat(values).detach()
            states = torch.cat(states)
            actions = torch.cat(actions)
            gae_returns = torch.cat(gae_returns).detach()
            mc_returns = torch.cat(mc_returns).detach()
            
            dist_probs = torch.cat(dist_probs).detach()

            ### Calculate advantage for the batch ###
            gae_advantages = gae_returns - values
            gae_advantages = self.normalize(gae_advantages)
            
            mc_advantages = mc_returns - values  # monte carlo advantage (debugging info)
            mc_advantages = self.normalize(mc_advantages)

            # print(f'actions: {actions}')
            
            # Calculate total reward of current batch (debugging info)
            epoch_total_reward = sum(rewards).item()
            
            # Calculate gamma (debugging info)
            sensor_gamma = self.calc_batch_gamma(taus)
            epoch_mean_gamma = np.mean(sensor_gamma)
            
            ### Write PPO Batch data to file ### ~ 0.005 s
            for step in range(self.ppo_steps):
                frame_idx += 1
                state_elements = [s.item() for s in states[step]]
                action_elements = [a.item() for a in actions[step]]

                # print(f'vol_flow[step]: {vol_flows[step]}')  # TEST
                # print(f'taus[step]: {taus[step]}')  # TEST

                ppo_batch_data = [frame_idx, train_epoch, step + 1, 
                                  epoch_total_reward, gae_returns[step].item(), 
                                  mc_returns[step].item(), values[step].item(), 
                                  rewards[step].item(), 
                                  gae_advantages[step].item(), 
                                  mc_advantages[step].item(), 
                                  act_log_probs[step].item(), 
                                  dist_probs[step].item()] + action_elements + state_elements + \
                                 [vol_flows[step]] + taus[step] + [epoch_mean_gamma] + sensor_gamma

                write_file(self.fname_batch, ppo_batch_data)

            ### Optimizing the policy and value network ###
            ppo_update_data = self.ppo_update(frame_idx, train_epoch, states, 
                                              actions, act_log_probs, 
                                              gae_returns, gae_advantages)
            
            ### Write PPO Update data to file ### ~ 0.002 s
            write_file(self.fname_update, ppo_update_data)
            
            ### Save model ###
            if train_epoch % self.save_interval == 0:
                torch.save(self.model.state_dict(), f'{self.chkpnt_dir}/epoch_{train_epoch}_torch_model')
                print("\n##### Model saved #####\n")
            
            print(f'\nTraining Epoch: {train_epoch} \nTotal Reward of current Batch: {sum(rewards).item()} \nBatchsize: {self.ppo_steps}\nSum loss actor: {ppo_update_data[2]}\nSum loss critc: {ppo_update_data[3]}\nSum loss total: {ppo_update_data[4]}\nSum entropy: {ppo_update_data[5]}\n')
            
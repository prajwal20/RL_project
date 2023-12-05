import numpy as np
import pandas as pd
import logging

def fileUpdate( model, slippery, learning_rate, horizon, steps, episode, cumulative_reward_history, win_history, mean_episode_length ):
    if learning_rate == 0.1:
        # Record Cumulative Reward History
        cumulative_reward_df = pd.DataFrame()
        cumulative_reward_df['Episode'] = np.arange(1, episode + 1)
        cumulative_reward_df['CumulativeReward'] = cumulative_reward_history
        cumulative_reward_df['Horizon'] = horizon
        cumulative_reward_df['Steps'] = steps
        cumulative_reward_df['Slippery'] = slippery
        cumulative_reward_df['Model'] = model

        cumulative_reward_file = pd.read_excel('CumulativeReward.xlsx')
        cumulative_reward_file = cumulative_reward_file[
                                                            (cumulative_reward_file['Horizon']!=horizon) |
                                                            (cumulative_reward_file['Steps']!=steps) |
                                                            (cumulative_reward_file['Slippery']!=slippery) |
                                                            (cumulative_reward_file['Model']!=model)
                                                        ].reset_index(drop=True)

        cumulative_reward_file = pd.concat([cumulative_reward_file, cumulative_reward_df], ignore_index=True)
        cumulative_reward_file.to_excel('CumulativeReward.xlsx', index=False)

        logging.info('Updated Cumulative Reward History')

        # Record Win Rate History
        win_history_df = pd.DataFrame(win_history, columns=['Episode', 'WinRate'])
        win_history_df['Horizon'] = horizon
        win_history_df['Steps'] = steps
        win_history_df['Slippery'] = slippery
        win_history_df['Model'] = model

        win_history_file = pd.read_excel('WinRate.xlsx')
        win_history_file = win_history_file[
                                                (win_history_file['Horizon']!=horizon) |
                                                (win_history_file['Steps']!=steps) |
                                                (win_history_file['Slippery']!=slippery) |
                                                (win_history_file['Model']!=model)
                                            ].reset_index(drop=True)

        win_history_file = pd.concat([win_history_file, win_history_df], ignore_index=True)
        win_history_file.to_excel('WinRate.xlsx', index=False)

        logging.info('Updated Win Rate History')

    # Record Mean Episode Length
    mean_episode_length_df = pd.DataFrame()
    mean_episode_length_df['Slippery'] = [slippery]
    mean_episode_length_df['Model'] = model
    mean_episode_length_df['Horizon'] = horizon
    mean_episode_length_df['Steps'] = steps
    mean_episode_length_df['Alpha'] = learning_rate
    mean_episode_length_df['MeanEpisodeLength'] = mean_episode_length

    mean_episode_length_file = pd.read_excel('MeanEpisodeLength.xlsx')
    mean_episode_length_file = mean_episode_length_file[
                                            (mean_episode_length_file['Horizon']!=horizon) |
                                            (mean_episode_length_file['Steps']!=steps) |
                                            (mean_episode_length_file['Slippery']!=slippery) |
                                            (mean_episode_length_file['Model']!=model) |
                                            (mean_episode_length_file['Alpha']!=learning_rate)
                                        ].reset_index(drop=True)

    mean_episode_length_file = pd.concat([mean_episode_length_file, mean_episode_length_df], ignore_index=True)
    mean_episode_length_file.to_excel('MeanEpisodeLength.xlsx', index=False)

    logging.info('Updated Mean Episode Length History')
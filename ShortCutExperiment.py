import time
import numpy as np
import pandas as pd

import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import subprocess

import torch
from transformers import set_seed
from ShortCutEnvironment import ShortcutEnvironment, WindyShortcutEnvironment
from ShortCutAgents import QLearningAgent, DynaAgent, DynaLLMAgent
from Helper import LearningCurvePlot, ComparisonPlot, smooth


seed = 42
set_seed(seed,deterministic=True)

#experiment parameters
agent_list = ["Dyna_LLM","Q-learning","Dyna_Q"]
env_list = [ShortcutEnvironment,WindyShortcutEnvironment]
n_repetitions = 40
llm_n_repetitions = 10
n_timesteps = 601
llm_n_timesteps = 301
eval_interval = 15
# per evaluation
n_eval_episodes = 30
max_episode_length = 100
# the windy environment has a 50% chance of wind downwards
# plotting
smoothing_window = 5
# RL models
epsilon_list = [0.1]
alpha_list = [0.1]
gamma = 1.0
# model-based
n_planning_updates = 10


# to track code running progress (sothat the user can see how much progress is made):
progress_counter = 0 # progress_counter +=1 every local repetition
total_progres_should_be = ((len(agent_list)-1)*len(env_list)*n_repetitions)+(len(env_list)*llm_n_repetitions)

# dictionaries that are later converted to dataframes and saved in the corresponding .csv file, saving the results of the experiment
dict_run_rep_r = {} # the reward the reward dict, first column is the proper index (the proper values of the x_axis in the plot)
dict_run_rep_time = {} # the time duration dict
dict_run_rep_dyna_r = {} # the rest of the dict's have the same structure as the previous two
dict_run_rep_dyna_time = {}
dict_run_rep_llm_r = {}
dict_run_rep_llm_time = {}


def print_greedy_actions(agent,environment,Q,r=12,c=12):
    greedy_actions = np.argmax(Q, 1).reshape((r,c))
    print_string = np.zeros((r, c), dtype=str)
    print_string[greedy_actions==0] = '^'
    print_string[greedy_actions==1] = 'v'
    print_string[greedy_actions==2] = '<'
    print_string[greedy_actions==3] = '>'
    print_string[np.max(Q, 1).reshape((r, c))==0] = '0'
    line_breaks = np.zeros((r,1), dtype=str)
    line_breaks[:] = '\n'
    print_string = np.hstack((print_string, line_breaks))
    # make the string to be saved
    string = f"Agent: {agent} in environment: {environment}, first time after {n_timesteps} timesteps:\n"
    clean_str = ''.join(map(str, print_string.flatten()))
    string += clean_str
    # string += str(print_string)
    with open("greedy_actions.txt", "a", encoding="utf-8") as text_file:
        text_file.write(string) # saving the "best" actions 


def experiment():
    # these global variables are changed in this function
    global dict_run_rep_r
    global dict_run_rep_time
    global dict_run_rep_dyna_r
    global dict_run_rep_dyna_time
    global dict_run_rep_llm_r
    global dict_run_rep_llm_time

    # print the two environments:
    for env in env_list:
        e = env() # Initialize 
        e.render() # saving the environment print in environment.txt

    # get the indexes (the x-axis) for all the reward dataframes for all algorithms except Dyna-LLM 
    if n_timesteps % eval_interval == 0: # if so, the correct amount is one less than the else case
        x = np.zeros(shape=((n_timesteps//eval_interval))) # x axis for every envaluation
    else:
        x = np.zeros(shape=((n_timesteps//eval_interval)+1)) # x axis for every envaluation
    for idx in range(len(x)):
        x[idx]=eval_interval*idx # calculate the corresponding t value
    dict_run_rep_r["indx"] = pd.Series(x) # make the first column "indx"
    dict_run_rep_dyna_r["indx"] = pd.Series(x)

    # get the indexes (the x-axis) for all the reward dataframes for Dyna-LLM
    if llm_n_timesteps % eval_interval == 0: # if so, the correct amount is one less than the else case
        x = np.zeros(shape=((llm_n_timesteps//eval_interval))) # x axis for every envaluation
    else:
        x = np.zeros(shape=((llm_n_timesteps//eval_interval)+1)) # x axis for every envaluation
    for idx in range(len(x)):
        x[idx]=eval_interval*idx # calculate the corresponding t value
    dict_run_rep_llm_r["indx"] = pd.Series(x)
    
    
    for agent in agent_list: # select the agent for the experiments
        for env in env_list: # select environment for the experiments
            if agent == "Dyna_LLM":
                run_repetitions(n_timesteps=llm_n_timesteps,eval_interval=eval_interval,n_repetitions=llm_n_repetitions,\
                                environment=env,epsilon_list= epsilon_list,alpha_list=alpha_list,agent=agent)
            else:
                run_repetitions(n_timesteps=n_timesteps,eval_interval=eval_interval,n_repetitions=n_repetitions,\
                                environment=env,epsilon_list= epsilon_list,alpha_list=alpha_list,agent=agent)
    
    
    # make the dictionaries dataframes
    if dict_run_rep_r:
        dict_run_rep_r = pd.concat(dict_run_rep_r, axis=1)
    else:
        dict_run_rep_r = pd.DataFrame()

    if dict_run_rep_time:
        dict_run_rep_time = pd.concat(dict_run_rep_time, axis=1)
    else:
        dict_run_rep_time = pd.DataFrame()
    
    if dict_run_rep_dyna_r:
        dict_run_rep_dyna_r = pd.concat(dict_run_rep_dyna_r, axis=1)
    else:
        dict_run_rep_dyna_r = pd.DataFrame()

    if dict_run_rep_dyna_time:
        dict_run_rep_dyna_time = pd.concat(dict_run_rep_dyna_time, axis=1)
    else:
        dict_run_rep_dyna_time = pd.DataFrame()

    if dict_run_rep_llm_r:
        dict_run_rep_llm_r = pd.concat(dict_run_rep_llm_r, axis=1)
    else:
        dict_run_rep_llm_r = pd.DataFrame()

    if dict_run_rep_llm_time:
        dict_run_rep_llm_time = pd.concat(dict_run_rep_llm_time, axis=1)
    else:
        dict_run_rep_llm_time = pd.DataFrame()

    
    #save dataframes in csv files
    dict_run_rep_r.to_csv("q_r.csv", header=True, index=False)
    dict_run_rep_time.to_csv("q_time.csv", header=True, index=False)
    dict_run_rep_dyna_r.to_csv("dyna_r.csv", header=True, index=False)
    dict_run_rep_dyna_time.to_csv("dyna_time.csv", header=True, index=False)
    dict_run_rep_llm_r.to_csv("llm_r.csv", header=True, index=False)
    dict_run_rep_llm_time.to_csv("llm_time.csv", header=True, index=False)

    # make plots with the csv files, executes the "plotresults.py" file 
    # this is done in order to be able to change the plots without needing to redo the whole experiment
    subprocess.run(["python", "plotresults.py"])

def run_repetitions(n_timesteps, eval_interval, n_repetitions, environment,\
                    epsilon_list,alpha_list,agent):
    
    global progress_counter

    for epsilon in epsilon_list: # init epsilon (in a loop for the possibility to explore epsilon more)
        for alpha in alpha_list: # for all values of hyperparameter alfa
            for local_rep in range(n_repetitions): # for all repetitions
                # init the result arrays
                if n_timesteps % eval_interval == 0: # if so, the correct amount is one less than the else case
                    y = np.zeros(shape=((n_timesteps//eval_interval))) # array in which we store the reward of each evaluation
                else:
                    y = np.zeros(shape=((n_timesteps//eval_interval)+1)) # array in which we store the reward of each evaluation
                time_array = np.zeros(shape=n_timesteps) # array in which we store the time duration of each time step
                
                env = environment() # Initialize 
                state = env.state() # init the state
                
                n_actions = env.action_size() # get the amount of actions from enviorment
                n_states = env.state_size() # get the amount of states from enviorment
                # check if one of the valid agents is given to this function, select corresponding agent.
                if agent == "Q-learning":
                    pi = QLearningAgent(n_states=n_states,n_actions=n_actions,
                                    learning_rate=alpha,epsilon=epsilon) # Initialize Agent
                elif agent == "Dyna_Q":
                    pi = DynaAgent(n_actions=n_actions, n_states=n_states,epsilon=epsilon,
                                    learning_rate=alpha,gamma=gamma) # Initialize Agent
                elif agent == "Dyna_LLM":
                    pi = DynaLLMAgent(n_actions=n_actions, n_states=n_states,epsilon=epsilon,
                                    learning_rate=alpha,gamma=gamma) # Initialize Agent
                else:
                    raise ValueError("No valid agent chosen!")

                for t in range(n_timesteps): # for each timestep
                    print("beginnin timestep")
                    begin = time.perf_counter_ns() # start timer in nanoseconds
                    a = pi.select_action(state) # select action
                    r = env.step(a) # sample reward
                    next_state = env.state() # get next state
                    # update the corresponding policy
                    if agent == "Q-learning":
                        pi.update(state, a, r, next_state) # update policy
                    elif agent == "Dyna_Q":
                        pi.update(state, a, r, next_state, done=env.done(), n_planning_updates=n_planning_updates) # update policy
                    else: #else, agent must be "Dyna_LLM"
                        pi.update(state, a, r, next_state, done=env.done(), n_planning_updates=n_planning_updates) # update policy
                    
                    if env.done(): # when goal is reached, reset environment
                        state = env.reset()
                    else: # else, continue with episode
                        state = next_state # update state (the next state is "next_state")
                    end = time.perf_counter_ns() # end timer
                    total = end - begin # calculate duration
                    time_array[t] = total # update the time in nanoseconds
                    if t%eval_interval==0: # evaluate when planned
                        r_mean = pi.evaluate(env,n_eval_episodes=n_eval_episodes, max_episode_length=max_episode_length)
                        y[t//eval_interval] = r_mean # update reward array

                        print(r_mean)
                    print(total)
                    print("end timestep")
                    
                
                # save results in the corresponding dataframes
                if agent == "Q-learning":
                    dict_run_rep_r[f"{environment}+{epsilon}+{alpha}+{local_rep}"] = pd.Series(y) #save the evaluation results
                    dict_run_rep_time[f"{environment}+{epsilon}+{alpha}+{local_rep}"] = pd.Series(time_array) #save the time results
                elif agent == "Dyna_Q":
                    dict_run_rep_dyna_r[f"{environment}+{epsilon}+{alpha}+{local_rep}"] = pd.Series(y) #save the evaluation results
                    dict_run_rep_dyna_time[f"{environment}+{epsilon}+{alpha}+{local_rep}"] = pd.Series(time_array) #save the time results
                else: #else, agent must be "Dyna_LLM"
                    dict_run_rep_llm_r[f"{environment}+{epsilon}+{alpha}+{local_rep}"] = pd.Series(y) #save the evaluation results
                    dict_run_rep_llm_time[f"{environment}+{epsilon}+{alpha}+{local_rep}"] = pd.Series(time_array) #save the time results
                
                # save first greedy actions in greedy_actions.txt
                if local_rep == 0:
                    print_greedy_actions(agent,environment,pi.Q,env.r,env.c) # print the map of actions corresponding to the Q values

                # show the user the progress of the experiment
                progress_counter += 1
                print(f"Progress: {progress_counter}/{total_progres_should_be}")

    print("done") # indicate that the function is done

    
if __name__ == '__main__':
    # run the experiment
    experiment()


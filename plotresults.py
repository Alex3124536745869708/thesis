import pandas as pd
import itertools
from Helper import LearningCurvePlot, ComparisonPlot, smooth
from ShortCutExperiment import smoothing_window,n_repetitions,llm_n_repetitions,alpha_list,epsilon_list,env_list,agent_list


def select_result_data(agent):
    """
    input: agent name (and .csv files for each data frame)

    returns: dataframes for reward and time
    """
    # select result data: reward
    if agent == "Q-learning":
        df_r = pd.read_csv("q_r.csv")
    elif agent == "Dyna_Q":
        df_r = pd.read_csv("dyna_r.csv")
    elif agent == "Dyna_LLM":
        df_r = pd.read_csv("llm_r.csv")
    else:
        raise ValueError("No valid agent given!")
    # select result data: time duration
    if agent == "Q-learning":
        df_time = pd.read_csv("q_time.csv")
    elif agent == "Dyna_Q":
        df_time = pd.read_csv("dyna_time.csv")
    else: #Dyna_LLM
        df_time = pd.read_csv("llm_time.csv")
    return df_r,df_time


def calculate_mean_std_min_max():
    """
    calculates: mean, std, min, max of every measured timestep and updates the .csv files.
    returns: updated .csv files and a comparison plot of every single agent (seperately) in both environments for the reward and time duration.
    """
    for agent in agent_list:
        
        LCTest = LearningCurvePlot(title=f"Both environments {agent} learning curve") # Initialize plot and give title
        LCTest.ax.set_xlabel("Timesteps") # Set x-axis label
        LCTest.ax.set_ylabel("Average evaluation reward") # Set y-axis label

        LCtime = LearningCurvePlot(title=f"Both environments {agent} time") # Initialize plot and give title
        LCtime.ax.set_xlabel("Timesteps") # Set x-axis label
        LCtime.ax.set_ylabel("Average time duration (seconds)") # Set y-axis label

        df_r, df_time = select_result_data(agent)

        counter = 0 # counter to determine the start column index
        for env in env_list:
            for epsilon in epsilon_list:
                for alpha in alpha_list:
                    # -----------for the reward dataframes-----------
                    # determine the selection of the dataframe, in order to get the mean and standard deviation
                    if agent == "Dyna_LLM":
                        col_length = llm_n_repetitions # amount of columns of one experiment of one alpha, epsilon and environment
                    else:    
                        col_length = n_repetitions # amount of columns of one experiment of one alpha, epsilon and environment
                    start_col_idx = (col_length*counter)+1
                    end_col_idx = start_col_idx+col_length

                    # calculate the mean, standard deviation, minimum and maximum per timestep
                    mean = df_r[df_r.columns[start_col_idx:end_col_idx]].mean(axis=1) # get the mean
                    df_r[f"{env},{epsilon},{alpha},mean"] = mean
                    std = df_r[df_r.columns[start_col_idx:end_col_idx]].std(axis=1)
                    df_r[f"{env},{epsilon},{alpha},std"] = std
                    # min and max are not used for these graphs
                    min = df_r[df_r.columns[start_col_idx:end_col_idx]].min(axis=1)
                    df_r[f"{env},{epsilon},{alpha},min"] = min
                    max = df_r[df_r.columns[start_col_idx:end_col_idx]].max(axis=1)
                    df_r[f"{env},{epsilon},{alpha},max"] = max

                    LCTest.add_curve(y=smooth(mean,window=smoothing_window),std=std,x=df_r["indx"],\
                                        label=f"env = {env}",annotation=False) # add reward curve to plot
                    
                    # -----------for the timeduration dataframes-----------
                    # determine the selection of the dataframe, in order to get the mean and standard deviation
                    start_col_idx -= 1 #because the time dataframes do not have an index column
                    end_col_idx -= 1 #because the time dataframes do not have an index column

                    # calculate the mean, standard deviation, minimum and maximum per timestep
                    mean = (df_time[df_time.columns[start_col_idx:end_col_idx]].mean(axis=1))/1000000000 # get the mean and convert from nanosecond to second, so the mean is in sec but the raw data is in nano sec
                    df_time[f"{env},{epsilon},{alpha},mean"] = mean
                    std = (df_time[df_time.columns[start_col_idx:end_col_idx]].std(axis=1))/1000000000 #convert from nanosecond to second
                    df_time[f"{env},{epsilon},{alpha},std"] = std
                    min = (df_time[df_time.columns[start_col_idx:end_col_idx]].min(axis=1))/1000000000 #convert from nanosecond to second
                    df_time[f"{env},{epsilon},{alpha},min"] = min
                    max = (df_time[df_time.columns[start_col_idx:end_col_idx]].max(axis=1))/1000000000 #convert from nanosecond to second
                    df_time[f"{env},{epsilon},{alpha},max"] = max

                    LCtime.add_curve(y=smooth(mean,window=smoothing_window),std=std,\
                                        label=f"env = {env}",annotation=False) # add time curve to plot
                    #update counter
                    counter+=1

        LCTest.save(name=f'Both_environments-{agent}reward.png') # save plot
        LCtime.save(name=f'Both_environments-{agent}time.png') # save plot

        # save dataframes with the calculated means, standard deviation, min and max :
        if agent == "Q-learning":
            df_r.to_csv("q_r.csv", header=True, index=False) # reward
            df_time.to_csv("q_time.csv", header=True, index=False) # time
        elif agent == "Dyna_Q":
            df_r.to_csv("dyna_r.csv", header=True, index=False) # reward
            df_time.to_csv("dyna_time.csv", header=True, index=False) # time
        else: #Dyna_LLM
            df_r.to_csv("llm_r.csv", header=True, index=False) # reward
            df_time.to_csv("llm_time.csv", header=True, index=False) # time


def individual_agent_env_graphs():
    """
    Return: graphs of one environment and agent, for all agents and evironments
    """
    for agent in agent_list:
        df_r, df_time = select_result_data(agent)
        for env in env_list:
            LCTest = LearningCurvePlot(title=f"{agent} learning curve") # Initialize plot and give title
            LCTest.ax.set_xlabel("Timesteps") # Set x-axis label
            LCTest.ax.set_ylabel("Average evaluation reward") # Set y-axis label

            LCtime = LearningCurvePlot(title=f"{agent} time") # Initialize plot and give title
            LCtime.ax.set_xlabel("Timesteps") # Set x-axis label
            LCtime.ax.set_ylabel("Average time duration (seconds)") # Set y-axis label

            for epsilon in epsilon_list:
                for alpha in alpha_list:
                    # -----------for the reward dataframes-----------
                    mean = df_r[f"{env},{epsilon},{alpha},mean"]
                    std = df_r[f"{env},{epsilon},{alpha},std"]
                    LCTest.add_curve(y=smooth(mean,window=smoothing_window),std=std,x=df_r["indx"],\
                                        label=f"env = {env}",annotation=False) # add reward curve to plot
                    
                    # -----------for the timeduration dataframes-----------
                    mean= df_time[f"{env},{epsilon},{alpha},mean"]
                    std = df_time[f"{env},{epsilon},{alpha},std"]
                    LCtime.add_curve(y=smooth(mean,window=smoothing_window),std=std,\
                                        label=f"env = {env}",annotation=False) # add time curve to plot

            LCTest.save(name=f'{env}-{agent}reward.png') # save plot
            LCtime.save(name=f'{env}-{agent}time.png') # save plot

        
def make_comparison_per_env_graphs():
    """
    Return: graphs of agent pairs per environment
    """
    agent_combinations = list(itertools.combinations(agent_list, 2)) # make all combinations of two agents for the comparison plots
    for env in env_list:
        for combination in agent_combinations:
            # init plots
            title = ""
            for i, agent in enumerate(combination):
                if i == len(combination)-1:
                    title += f"{agent}"
                else: 
                    title+=f"{agent} and "
            LCTest = LearningCurvePlot(title=f"{title} learning curves") # Initialize plot and give title
            LCTest.ax.set_xlabel("Timesteps") # Set x-axis label
            LCTest.ax.set_ylabel("Average evaluation reward") # Set y-axis label
            LCtime = LearningCurvePlot(title=f"{title} time") # Initialize plot and give title
            LCtime.ax.set_xlabel("Timesteps") # Set x-axis label
            LCtime.ax.set_ylabel("Average time duration (seconds)") # Set y-axis label

            for agent in combination:
                df_r, df_time = select_result_data(agent)
                for epsilon in epsilon_list:
                    for alpha in alpha_list:
                        # -----------for the reward dataframes-----------
                        mean = df_r[f"{env},{epsilon},{alpha},mean"]
                        std = df_r[f"{env},{epsilon},{alpha},std"]
                        LCTest.add_curve(y=smooth(mean,window=smoothing_window),std=std,x=df_r["indx"],\
                                            label=f"agent = {agent}",annotation=False) # add reward curve to plot
                        
                        # -----------for the timeduration dataframes-----------
                        mean= df_time[f"{env},{epsilon},{alpha},mean"]
                        std = df_time[f"{env},{epsilon},{alpha},std"]
                        LCtime.add_curve(y=smooth(mean,window=smoothing_window),std=std,\
                                            label=f"agent = {agent}",annotation=False) # add time curve to plot

            LCTest.save(name=f'{env}-{title}reward.png') # save plot
            LCtime.save(name=f'{env}-{title}time.png') # save plot



def mean_min_max_individual():
    """
    Return: graphs with min and max instead of std for every agent seperately
    """
    for agent in agent_list:
        
        LCTest = LearningCurvePlot(title=f"Both environments {agent} learning curve") # Initialize plot and give title
        LCTest.ax.set_xlabel("Timesteps") # Set x-axis label
        LCTest.ax.set_ylabel("Average evaluation reward") # Set y-axis label

        LCtime = LearningCurvePlot(title=f"Both environments {agent} time") # Initialize plot and give title
        LCtime.ax.set_xlabel("Timesteps") # Set x-axis label
        LCtime.ax.set_ylabel("Average time duration (seconds)") # Set y-axis label

        df_r, df_time = select_result_data(agent)

        for env in env_list:
            for epsilon in epsilon_list:
                for alpha in alpha_list:
                    # -----------for the reward dataframes-----------
                    # calculate the mean, standard deviation, minimum and maximum per timestep
                    mean = df_r[f"{env},{epsilon},{alpha},mean"]
                    min = df_r[f"{env},{epsilon},{alpha},min"]
                    max = df_r[f"{env},{epsilon},{alpha},max"]

                    LCTest.add_curve(y=smooth(mean,window=smoothing_window),x=df_r["indx"],min=min,max=max,\
                                        label=f"env = {env}",annotation=False) # add reward curve to plot
                    
                    # -----------for the timeduration dataframes-----------
                    # calculate the mean, standard deviation, minimum and maximum per timestep
                    mean = df_time[f"{env},{epsilon},{alpha},mean"]
                    min = df_time[f"{env},{epsilon},{alpha},min"]
                    max = df_time[f"{env},{epsilon},{alpha},max"]

                    LCtime.add_curve(y=smooth(mean,window=smoothing_window),min=min,max=max,\
                                        label=f"env = {env}",annotation=False) # add time curve to plot
                    

        LCTest.save(name=f'minmax_individual-{agent}reward.png') # save plot
        LCtime.save(name=f'minmax_individual-{agent}time.png') # save plot


def all_mm_nowind_vs():
    """
    Return: NoWind comparrisson graph for all algorithms, min max
    """
    LCTest = LearningCurvePlot(title=f"All agents learning curve in NoWind env") # Initialize plot and give title
    LCTest.ax.set_xlabel("Timesteps") # Set x-axis label
    LCTest.ax.set_ylabel("Average evaluation reward") # Set y-axis label

    LCtime = LearningCurvePlot(title=f"All agents time in NoWind env") # Initialize plot and give title
    LCtime.ax.set_xlabel("Timesteps") # Set x-axis label
    LCtime.ax.set_ylabel("Average time duration (seconds)") # Set y-axis label

    env = "NoWind"

    for agent in agent_list:
        df_r, df_time = select_result_data(agent)

        
        for epsilon in epsilon_list:
            for alpha in alpha_list:
                # -----------for the reward dataframes-----------
                # calculate the mean, standard deviation, minimum and maximum per timestep
                mean = df_r[f"{env},{epsilon},{alpha},mean"]
                min = df_r[f"{env},{epsilon},{alpha},min"]
                max = df_r[f"{env},{epsilon},{alpha},max"]

                LCTest.add_curve(y=smooth(mean,window=smoothing_window),x=df_r["indx"],min=min,max=max,\
                                    label=f"agent = {agent}",annotation=False) # add reward curve to plot
                
                # -----------for the timeduration dataframes-----------
                # calculate the mean, standard deviation, minimum and maximum per timestep
                mean = df_time[f"{env},{epsilon},{alpha},mean"]
                min = df_time[f"{env},{epsilon},{alpha},min"]
                max = df_time[f"{env},{epsilon},{alpha},max"]

                LCtime.add_curve(y=smooth(mean,window=smoothing_window),min=min,max=max,\
                                    label=f"agent = {agent}",annotation=False) # add time curve to plot
                

        LCTest.save(name=f'allmmNoWindreward.png') # save plot
        LCtime.save(name=f'allmmNoWindtime.png') # save plot


def all_mm_windy_vs():
    """
    Return: NoWind comparrisson graph for all algorithms, min max
    """
    LCTest = LearningCurvePlot(title=f"All agents learning curve in Windy env") # Initialize plot and give title
    LCTest.ax.set_xlabel("Timesteps") # Set x-axis label
    LCTest.ax.set_ylabel("Average evaluation reward") # Set y-axis label

    LCtime = LearningCurvePlot(title=f"All agents time in Windy") # Initialize plot and give title
    LCtime.ax.set_xlabel("Timesteps") # Set x-axis label
    LCtime.ax.set_ylabel("Average time duration (seconds)") # Set y-axis label

    env = "Windy"

    for agent in agent_list:
        df_r, df_time = select_result_data(agent)

        
        for epsilon in epsilon_list:
            for alpha in alpha_list:
                # -----------for the reward dataframes-----------
                # calculate the mean, standard deviation, minimum and maximum per timestep
                mean = df_r[f"{env},{epsilon},{alpha},mean"]
                min = df_r[f"{env},{epsilon},{alpha},min"]
                max = df_r[f"{env},{epsilon},{alpha},max"]

                LCTest.add_curve(y=smooth(mean,window=smoothing_window),x=df_r["indx"],min=min,max=max,\
                                    label=f"agent = {agent}",annotation=False) # add reward curve to plot
                
                # -----------for the timeduration dataframes-----------
                # calculate the mean, standard deviation, minimum and maximum per timestep
                mean = df_time[f"{env},{epsilon},{alpha},mean"]
                min = df_time[f"{env},{epsilon},{alpha},min"]
                max = df_time[f"{env},{epsilon},{alpha},max"]

                LCtime.add_curve(y=smooth(mean,window=smoothing_window),min=min,max=max,\
                                    label=f"agent = {agent}",annotation=False) # add time curve to plot
                

        LCTest.save(name=f'allmmWindyreward.png') # save plot
        LCtime.save(name=f'allmmWindytime.png') # save plot


def all_std_nowind_vs():
    """
    Return: NoWind comparrisson graph for all algorithms, std
    """
    LCTest = LearningCurvePlot(title=f"All agents learning curve in NoWind env") # Initialize plot and give title
    LCTest.ax.set_xlabel("Timesteps") # Set x-axis label
    LCTest.ax.set_ylabel("Average evaluation reward") # Set y-axis label

    LCtime = LearningCurvePlot(title=f"All agents time in NoWind env") # Initialize plot and give title
    LCtime.ax.set_xlabel("Timesteps") # Set x-axis label
    LCtime.ax.set_ylabel("Average time duration (seconds)") # Set y-axis label

    env = "NoWind"

    for agent in agent_list:
        df_r, df_time = select_result_data(agent)

        
        for epsilon in epsilon_list:
            for alpha in alpha_list:
                # -----------for the reward dataframes-----------
                # calculate the mean, standard deviation, minimum and maximum per timestep
                mean = df_r[f"{env},{epsilon},{alpha},mean"]
                std = df_r[f"{env},{epsilon},{alpha},std"]

                LCTest.add_curve(y=smooth(mean,window=smoothing_window),x=df_r["indx"],std=std,\
                                    label=f"agent = {agent}",annotation=False) # add reward curve to plot
                
                # -----------for the timeduration dataframes-----------
                # calculate the mean, standard deviation, minimum and maximum per timestep
                mean = df_time[f"{env},{epsilon},{alpha},mean"]
                std = df_time[f"{env},{epsilon},{alpha},std"]

                LCtime.add_curve(y=smooth(mean,window=smoothing_window),std=std,\
                                    label=f"agent = {agent}",annotation=False) # add time curve to plot
                

        LCTest.save(name=f'allstdNoWindreward.png') # save plot
        LCtime.save(name=f'allstdNoWindtime.png') # save plot


def all_std_windy_vs():
    """
    Return: NoWind comparrisson graph for all algorithms, std
    """
    LCTest = LearningCurvePlot(title=f"All agents learning curve in Windy env") # Initialize plot and give title
    LCTest.ax.set_xlabel("Timesteps") # Set x-axis label
    LCTest.ax.set_ylabel("Average evaluation reward") # Set y-axis label

    LCtime = LearningCurvePlot(title=f"All agents time in Windy") # Initialize plot and give title
    LCtime.ax.set_xlabel("Timesteps") # Set x-axis label
    LCtime.ax.set_ylabel("Average time duration (seconds)") # Set y-axis label

    env = "Windy"

    for agent in agent_list:
        df_r, df_time = select_result_data(agent)

        
        for epsilon in epsilon_list:
            for alpha in alpha_list:
                # -----------for the reward dataframes-----------
                # calculate the mean, standard deviation, minimum and maximum per timestep
                mean = df_r[f"{env},{epsilon},{alpha},mean"]
                std = df_r[f"{env},{epsilon},{alpha},std"]

                LCTest.add_curve(y=smooth(mean,window=smoothing_window),x=df_r["indx"],std=std,\
                                    label=f"agent = {agent}",annotation=False) # add reward curve to plot
                
                # -----------for the timeduration dataframes-----------
                # calculate the mean, standard deviation, minimum and maximum per timestep
                mean = df_time[f"{env},{epsilon},{alpha},mean"]
                std = df_time[f"{env},{epsilon},{alpha},std"]

                LCtime.add_curve(y=smooth(mean,window=smoothing_window),std=std,\
                                    label=f"agent = {agent}",annotation=False) # add time curve to plot
                

        LCTest.save(name=f'allstdWindyreward.png') # save plot
        LCtime.save(name=f'allstdWindytime.png') # save plot


def get_average_average_timeduration():
    """
    Print in the terminal: Means of mean time durations from the main experiment.
    """
    print("The means of the mean time durations:")
    for agent in agent_list:
        _, df_time = select_result_data(agent)
        for env in env_list:
            for epsilon in epsilon_list:
                for alpha in alpha_list:
                    print(f"agent: {agent}, env: {env}")
                    mean_of_means = df_time[f"{env},{epsilon},{alpha},mean"].mean(axis=0) # mean of the column
                    print(f"{mean_of_means} seconds")
                    

# make the evironment into shorter strings for naming and check if the envs are valid
for env in env_list:
    if str(env) == "<class 'ShortCutEnvironment.ShortcutEnvironment'>":
        pass
    elif str(env) == "<class 'ShortCutEnvironment.WindyShortcutEnvironment'>":
        pass
    else:
        raise ValueError(f"{env} is an invalid environment for the plotting of graphs. Possibly update the make_graphs function.")
env_list = ["NoWind","Windy"]

# make graphs
calculate_mean_std_min_max()
individual_agent_env_graphs()
make_comparison_per_env_graphs()
mean_min_max_individual()
all_mm_nowind_vs()
all_mm_windy_vs()
all_std_nowind_vs()
all_std_windy_vs()

# print all averages of average time duration for all agents in all environments
get_average_average_timeduration()



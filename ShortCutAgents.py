import random
import numpy as np
import copy
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



class QLearningAgent(object):

    def __init__(self, n_states, n_actions,learning_rate,epsilon):
        self.n_actions = n_actions
        self.n_states = n_states
        self.epsilon = epsilon
        # Init Q value table, alfa, gamma
        self.Q = np.zeros((n_states,n_actions))
        self.alpha = learning_rate
        self.gamma = 1
        pass
        
    def select_action(self, state):
        if state is None :
            raise ValueError('State must not be None, look at environment init')
        #chance of best action
        p = 1 - self.epsilon
        # if the max action is not selected,
        if np.random.binomial(1, p) == 0:
            a = np.random.choice(self.n_actions) # then choose a random action
        else:
            # get the index of the max action for the state without bias
            a = np.random.choice(np.where(self.Q[state] == np.max(self.Q[state]))[0])
        
        return a
        
    def update(self, state, action, reward, next_state):
        # Update formula: s' = next state
        # max_A = the action with the maximum value (=A) is filled in
        # Q(s,a) = Q(s,a) + alfa * (r + gamma * max_A(Q(s', A)) - Q(s,a))
        self.Q[state, action] = self.Q[state, action] + self.alpha * \
            (reward + self.gamma * np.max(self.Q[next_state]) - \
            self.Q[state, action])
        pass

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        eval_env = copy.deepcopy(eval_env) # make a deep copy of the env, in order to make steps in evaluate that do not change the current state of the agent
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = np.random.choice(np.where(self.Q[s] == np.max(self.Q[s]))[0]) # fair greedy action selection
                r = eval_env.step(a)                
                s_next = eval_env.state()
                R_ep += r
                if eval_env.done():
                    break
                else:
                    s = s_next
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return


class DynaAgent:

    def __init__(self, n_states, n_actions, learning_rate=0.1, epsilon=0.1, gamma=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros((n_states,n_actions))
        # Initialize count tables, and reward sum tables. 
        self.n_sas = np.zeros((n_states,n_actions,n_states)) # counter of ending up at s_next from s by performing action a
        self.r_sum_sas = np.zeros((n_states,n_actions,n_states)) # total reward
        self.p_hat = np.zeros((n_states,n_actions,n_states)) # transition function
        self.r_hat = np.zeros((n_states,n_actions,n_states)) # average reward of s_next from s with a

    def select_action(self, s):
        #chance of best action
        p = 1 - self.epsilon
        # if the max action is not to be selected,
        if np.random.binomial(1, p) == 0:
            # then choose a random action that is not the best action
            a = np.random.choice(self.n_actions)
        else:
            # get the index of the max action for the state
            a = np.random.choice(np.where(self.Q[s] == np.max(self.Q[s]))[0])

        return a
        
    def update(self,s,a,r,s_next,done=False,n_planning_updates=0):
        #update model
        self.n_sas[s,a,s_next]+=1 # counter of ending up at s_next from a by performing action a
        self.r_sum_sas[s,a,s_next]+=r # total reward
        self.p_hat[s,a,s_next] = self.n_sas[s,a,s_next]/np.sum(self.n_sas[s,a,:]) # estimate transition function
        self.r_hat[s,a,s_next] = self.r_sum_sas[s,a,s_next]/self.n_sas[s,a,s_next] # estimate reward function
        
        # update Q-table:
        # s' = s_next
        # max_A = the action with the maximum value (=A) is filled in
        # Q(s,a) = Q(s,a) + alfa * (r + gamma * max_A(Q(s', A)) - Q(s,a))
        new = self.learning_rate * \
            (r + self.gamma * np.max(self.Q[s_next]) - \
            self.Q[s, a])

        self.Q[s, a] = self.Q[s, a] + new

        if done: # stop if the goal is achieved, maybe somewhere else!?
            return

        for _ in range(n_planning_updates):
            # s and a where n_sas_prime is not 0
            s = np.random.choice(np.nonzero(self.n_sas)[0]) # state to plan on
            a = np.random.choice(np.nonzero(self.n_sas[s])[0]) # planning action
            
            s_next = np.random.choice(range(len(self.p_hat[s,a])), p=self.p_hat[s,a]/np.sum(self.p_hat[s,a,:])) # simulate model
            r = self.r_hat[s,a,s_next] # get mean reward

            # update Q-table
            self.Q[s, a] = self.Q[s, a] + self.learning_rate * \
            (r + self.gamma * np.max(self.Q[s_next]) - \
            self.Q[s, a])
         
        pass

    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        eval_env = copy.deepcopy(eval_env) # make a deep copy of the env, in order to make steps in evaluate that do not change the current state of the agent
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = np.random.choice(np.where(self.Q[s] == np.max(self.Q[s]))[0]) # fair greedy action selection
                r = eval_env.step(a)                
                s_next = eval_env.state()
                R_ep += r
                if eval_env.done():
                    break
                else:
                    s = s_next
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return

class DynaLLMAgent:

    def __init__(self, n_states, n_actions, learning_rate=0.1, epsilon=0.1, gamma=1.0):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.gamma = gamma
        self.Q = np.zeros((n_states,n_actions))
        self.n_sas = np.zeros((n_states,n_actions,n_states)) # counter of ending up at s_next from s by performing action a
        # init llm model
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu" #  print dit om te checken wat de device is
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        self.history_prompt = ""

    def select_action(self, s):
        #chance of best action
        p = 1 - self.epsilon
        # if the max action is not to be selected,
        if np.random.binomial(1, p) == 0:
            # then choose a random action that is not the best action
            a = np.random.choice(self.n_actions)
        else:
            # get the index of the max action for the state
            a = np.random.choice(np.where(self.Q[s] == np.max(self.Q[s]))[0])

        return a
        

    @staticmethod
    def action_to_string(a):
        """
        Give natural language as output as string for the corresponding action, previously represented as integer.
        Input: action as integer between from 0 to 3
        Output: action as string
        """
        if isinstance(a, np.int64) or isinstance(a, int):
            if a == 0:
                action = "up"
            elif a == 1:
                action = "down"
            elif a == 2:
                action = "left"
            elif a == 3:
                action = "right"
            else:
                raise ValueError(f"The input variable must be between 0 and 3, the input variable has value {action}.")
            return action
        else:
            atype = type(a)
            raise TypeError(f"The input variable is supposed to be an int, it is an {atype}.")
    
    @staticmethod
    def check_output_of_llm(input_text,output_text):
        """
        Checking if output of llm has the correct synthax and giving the r and s_next from the promt and validity of the output.
        
        input: input text: string, output text: string

        output: r: int, s_next: int, invallid_output: Bool
        """
        # the output of the function when the llm output is invalid
        r=0
        s_next=0
        invalid_output =False
        # init the rest
        beginning_error = "is equal"
        input_text_length = len(input_text)

        # check beginnings
        # input_text_length-1 is done because in the eventual prompt the "." at the end becomes a " →" which is harmless.
        if output_text[:input_text_length-1] != input_text[:-1]:
            beginning_error = "is not equal"
            # check manually after running if the file: "LLM_output__validation_errors.txt" exists to see if the output is not valid. Since the beginnings being not equal does not mean that the answer is invalid, but it is an indication that it might be
            with open("LLM_output_validation_errors.txt", "a", encoding="utf-8") as text_file: # saving the error for better context
                text_file.write(f"##############################\nERROR:Beginnings are not equal\n------------------------------\ninput_text:\n\n{input_text}\n\n------------------------------\noutput_text:\n\n{output_text}\n\n")

        answer = output_text[input_text_length:] # get answer of llm
        # check if the answer has the right synthax, good answer synthax  example: # Response: "State: 21, Agent takes action: Agent moves up, Reward: -1, Next state: 16."
        right_synthax = re.search(r'Response: "State: \d+, Action: Agent moves (up|down|left|right), Reward: -?\d+, Next state: \d+\."', answer)
        try: # if it has the right synthax, continue
            answer = answer[right_synthax.start():right_synthax.end()] # keep only the right part of the answer
        except:
            # if the answer does not have the right synthax then skip the retrieval of r and s_next
            invalid_output = True
            with open("LLM_output_validation_errors.txt", "a", encoding="utf-8") as text_file: # saving the error for better context
                text_file.write(f"##############################\nERROR:Beginning {beginning_error}, Answer is not of right synthax\n------------------------------\ninput_text:\n\n{input_text}\n\n------------------------------\noutput_text:\n\n{output_text}\n\n------------------------------\nanswer:\n\n{answer}\n\n")
            return r, s_next, invalid_output
        
        # answer has correct synthax, get reward and next state from string
        answer=answer[:-2] # remove end of string
        answer_list = re.split(",",answer) # make list with s,a,r,s_next and excess
        r = int(answer_list[2][9:]) # remove excess (a way to get start_index: reward = int(answer_list[2][start_idx:]) # start_idx= len(" Reward: "), so start_idx=9)
        s_next = int(answer_list[3][13:])
        # for manual inspection of the r and s_next in case the beginnings were not equal
        if beginning_error == "is not equal":
            with open("LLM_output_validation_errors.txt", "a", encoding="utf-8") as text_file: # saving the error for better context
                text_file.write(f"\n\n------------------------------\nanswer:\n\n{answer}\n\nr: {r}, s_next: {s_next}\n------------------------------\n")

        return r, s_next, invalid_output
    

    def update(self,s,a,r,s_next,done=False,n_planning_updates=0):
        # update history prompt for llm
        action = self.action_to_string(a)
        self.history_prompt += """
State: """+str(s)+""", Action: Agent moves """+action+""" →
Response: "State: """+str(s)+""", Action: Agent moves """+action+""", Reward: """+str(r)+""", Next state: """+str(s_next)+"""."
"""     
        # max 37 states and responses: 37*3 is 111, so max 111 enters in self.history string
        enter_amount = self.history_prompt.count("\n")
        diff_enter_amount = enter_amount - 111
        if diff_enter_amount > 0: # if too long history prompt: remove the oldest history (remove the beginning)
            self.history_prompt = "".join(self.history_prompt.splitlines(True)[diff_enter_amount:])

        history_copy = self.history_prompt # deep copy for during the planning updates
        

        #update non llm model
        self.n_sas[s,a,s_next]+=1 # counter of ending up at s_next from a by performing action a

        # update Q-table:
        # s' = s_next
        # max_A = the action with the maximum value (=A) is filled in
        # Q(s,a) = Q(s,a) + alfa * (r + gamma * max_A(Q(s', A)) - Q(s,a))
        new = self.learning_rate * \
            (r + self.gamma * np.max(self.Q[s_next]) - \
            self.Q[s, a])

        self.Q[s, a] = self.Q[s, a] + new

        if done: # stop if the goal is achieved, maybe somewhere else!?
            return

        i=0 # the planning update loop counter
        while i < n_planning_updates: # for n_planning updates, however every time the answer is invalid do one extra loop

            # s and a where n_sas_prime is not 0
            s = np.random.choice(np.nonzero(self.n_sas)[0]) # state to plan on
            a = np.random.choice(np.nonzero(self.n_sas[s])[0]) # planning action
            # get llm output
            action = self.action_to_string(a) # action as string for prompt
            input_prompt=["""Role:
You are an AI that simulates an environment for model based reinforcement learning.
          
Environment Rules:
The agent is in a 5×5 grid maze with states numbered from 1 to 25, structured as follows:
Top-left corner: State 1.
Top-right corner: State 5.
Bottom-left corner: State 21.
Bottom-right corner: State 25.
The agent moves up, down, left, or right, but cannot move outside the grid.
Each move results in a reward of -1 unless otherwise specified.
Use past interactions if available.

If the action is new, apply logical reasoning based on the grid structure:   
Moving left decreases the state number by 1, unless at the left boundary.
Moving right increases the state number by 1, unless at the right boundary.
Moving up decreases the state by 5, unless at the top boundary.
Moving down increases the state by 5, unless at the bottom boundary.
Circumstance at the beginning of the line means that YOU should reason with the effect the wind could have on the outcome of the action.

You must return the following after an action:
state, location, action, reward, next_state, next_location.
                    
Past Interactions:
    Context: The action from a certain state might result in a different next_state than the following action because of environment circumstanses.
    In case this happens, write: "Circumstance" in the output.
"""+history_copy+"""
Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: """+str(s)+""", Action: Agent moves """+action+"""."""]
            # Tokenize input
            encoded_input = self.tokenizer(input_prompt, return_tensors='pt',padding=True, truncation=True).to(self.device)
            input_length = len(encoded_input["input_ids"][0])
            length_of_answer = 45 # given my expected output
            buffer_length = 5
            # Generate text
            with torch.no_grad():
                output_tokens = self.model.generate(**encoded_input, max_length=round(input_length+length_of_answer+buffer_length)).to(self.device)
            # Decode the generated text
            generated_text = self.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            # checking synthax of llm output, get s_next from generated_text, get r from generated_text
            r,s_next,invalid_answer = self.check_output_of_llm(input_prompt[0],generated_text)
            

            # if the output of the llm is invallid, skip the Q-table update and loop counter update
            if invalid_answer:
                continue
            try: #incase the answer is still invalid use try,except (prevents index out of bounds error)
                # update Q-table
                self.Q[s, a] = self.Q[s, a] + self.learning_rate * \
                (r + self.gamma * np.max(self.Q[s_next]) - \
                self.Q[s, a])
            except:
                continue
            #update planning update loop counter
            i+=1
         
        pass
    


    def evaluate(self,eval_env,n_eval_episodes=30, max_episode_length=100):
        returns = []  # list to store the reward per episode
        eval_env = copy.deepcopy(eval_env) # make a deep copy of the env, in order to make steps in evaluate that do not change the current state of the agent
        for _ in range(n_eval_episodes):
            s = eval_env.reset()
            R_ep = 0
            for _ in range(max_episode_length):
                a = np.random.choice(np.where(self.Q[s] == np.max(self.Q[s]))[0]) # fair greedy action selection
                r = eval_env.step(a)                
                s_next = eval_env.state()
                R_ep += r
                if eval_env.done():
                    break
                else:
                    s = s_next
            returns.append(R_ep)
        mean_return = np.mean(returns)
        return mean_return
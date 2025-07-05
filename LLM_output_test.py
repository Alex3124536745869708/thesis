from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
seed = 42
set_seed(seed,deterministic=True)
import torch

def tinyllama(text):
    # init variables
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    

    # Tokenize input
    encoded_input = tokenizer(text, return_tensors='pt',padding=True, truncation=True).to(device)
    input_length = len(encoded_input["input_ids"][0])
    length_of_answer = 45 # given my expected output
    buffer_length = 5 # extra answer length to hopefully never cut off answers that are unexpected.

    # Generate text
    with torch.no_grad():
        output_tokens = model.generate(**encoded_input, max_length=round(input_length+length_of_answer+buffer_length)).to(device)

    # Decode and print the generated text
    generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    # only print the additional part of the output (= the answer), thus removing the prompt from the output
    input_text_length = len(text[0])
    answer = generated_text[input_text_length:] # get only the answer of llm
    print("Output of Tiny-Llama:")
    print(answer)
    print()

if __name__ == '__main__':

    # Prompt 3. old prompt without wind, featuring an known state-action combination of a known current state (if it is a known state-action combination then it has to be a known current state).
    print("Prompt 3.")
    text = ["""Role:
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves down →
Response: "State: 6, Action: Agent moves down, Reward: -1, Next state: 6."

State: 6, Action: Agent moves left →
Response: "State: 6, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves up →
Response: "State: 6, Action: Agent moves up, Reward: -1, Next state: 1."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 15, Action: Agent moves up.
"""]
    tinyllama(text) # the answer is correct

    # Prompt 4. old prompt without wind, featuring an unknown state-action combination of a known current state.
    print("Prompt 4.")
    text = ["""Role:
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves down →
Response: "State: 6, Action: Agent moves down, Reward: -1, Next state: 6."

State: 6, Action: Agent moves left →
Response: "State: 6, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves up →
Response: "State: 6, Action: Agent moves up, Reward: -1, Next state: 1."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 15, Action: Agent moves right.
"""]
    tinyllama(text) # the answer is incorrect

    # Prompt 5. old prompt without wind, featuring an unknown state-action combination of an unknown current state.
    print("Prompt 5.")
    text = ["""Role:
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves down →
Response: "State: 6, Action: Agent moves down, Reward: -1, Next state: 6."

State: 6, Action: Agent moves left →
Response: "State: 6, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves up →
Response: "State: 6, Action: Agent moves up, Reward: -1, Next state: 1."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 20, Action: Agent moves up.
"""]
    tinyllama(text) # the answer is incorrect

    # Prompt 6. old prompt with wind, featuring an known state-action combination
    print("Prompt 6.")
    text = ["""Role:
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves right →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 21."

State: 21, Action: Agent moves left →
Response: "State: 21, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves right →
Response: "State: 15, Action: Agent moves right, Reward: -1, Next state: 16."

State: 16, Action: Agent moves left →
Response: "State: 16, Action: Agent moves left, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 6."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 10, Action: Agent moves up.
"""]
    tinyllama(text) # the answer is correct

    # Prompt 6.5. bonus prompt, old prompt with wind, featuring an known state-action combination
    print("Prompt 6.5.")
    text = ["""Role:
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves right →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 21."

State: 21, Action: Agent moves left →
Response: "State: 21, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves right →
Response: "State: 15, Action: Agent moves right, Reward: -1, Next state: 16."

State: 16, Action: Agent moves left →
Response: "State: 16, Action: Agent moves left, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 6."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 10, Action: Agent moves right.
"""]
    tinyllama(text) # the answer is correct

    # Prompt 7. old prompt with wind, featuring an unknown state-action combination of a known current state.
    print("Prompt 7.")
    text = ["""Role:
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves right →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 21."

State: 21, Action: Agent moves left →
Response: "State: 21, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves right →
Response: "State: 15, Action: Agent moves right, Reward: -1, Next state: 16."

State: 16, Action: Agent moves left →
Response: "State: 16, Action: Agent moves left, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 6."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 15, Action: Agent moves down.
"""]
    tinyllama(text) # the answer is incorrect

    # Prompt 8. old prompt with wind, featuring an unknown state-action combination of a unknown current state.
    print("Prompt 8.")
    text = ["""Role:
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves right →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 21."

State: 21, Action: Agent moves left →
Response: "State: 21, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves right →
Response: "State: 15, Action: Agent moves right, Reward: -1, Next state: 16."

State: 16, Action: Agent moves left →
Response: "State: 16, Action: Agent moves left, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 6."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 13, Action: Agent moves down.
"""]
    tinyllama(text) # the answer is incorrect

    print("##########################New prompts######################################") #divide old and new prompt answers 

    # prompt 9. new prompt without wind, featuring an known state-action combination of a known current state (if it is a known state-action combination then it has to be a known current state).
    print("Prompt 9.")
    text = ["""Role:
You are an AI that simulates an environment for model based reinforcement learning.
          
Environment Rules:
The agent is in a 5×5 grid maze with states numbered from 0 to 24, structured as follows:
Top-left corner: State 0.
Top-right corner: State 4.
Bottom-left corner: State 20.
Bottom-right corner: State 24.
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves down →
Response: "State: 6, Action: Agent moves down, Reward: -1, Next state: 6."

State: 6, Action: Agent moves left →
Response: "State: 6, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves up →
Response: "State: 6, Action: Agent moves up, Reward: -1, Next state: 1."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 15, Action: Agent moves up.
"""]
    tinyllama(text) # the answer is correct

    # Prompt 10. new prompt without wind, featuring an unknown state-action combination of a known current state.
    print("Prompt 10.")
    text = ["""Role:
You are an AI that simulates an environment for model based reinforcement learning.
          
Environment Rules:
The agent is in a 5×5 grid maze with states numbered from 0 to 24, structured as follows:
Top-left corner: State 0.
Top-right corner: State 4.
Bottom-left corner: State 20.
Bottom-right corner: State 24.
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves down →
Response: "State: 6, Action: Agent moves down, Reward: -1, Next state: 6."

State: 6, Action: Agent moves left →
Response: "State: 6, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves up →
Response: "State: 6, Action: Agent moves up, Reward: -1, Next state: 1."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 15, Action: Agent moves right.
"""]
    tinyllama(text) # the answer is incorrect
    
    # Prompt 11. new prompt without wind, featuring an unknown state-action combination of an unknown current state.
    print("Prompt 11.")
    text = ["""Role:
You are an AI that simulates an environment for model based reinforcement learning.
          
Environment Rules:
The agent is in a 5×5 grid maze with states numbered from 0 to 24, structured as follows:
Top-left corner: State 0.
Top-right corner: State 4.
Bottom-left corner: State 20.
Bottom-right corner: State 24.
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves down →
Response: "State: 6, Action: Agent moves down, Reward: -1, Next state: 6."

State: 6, Action: Agent moves left →
Response: "State: 6, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 6."

State: 6, Action: Agent moves up →
Response: "State: 6, Action: Agent moves up, Reward: -1, Next state: 1."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 20, Action: Agent moves up.
"""]
    tinyllama(text) # the answer is incorrect

    # Prompt 12. new prompt with wind, featuring an known state-action combination
    print("Prompt 12.")
    text = ["""Role:
You are an AI that simulates an environment for model based reinforcement learning.
          
Environment Rules:
The agent is in a 5×5 grid maze with states numbered from 0 to 24, structured as follows:
Top-left corner: State 0.
Top-right corner: State 4.
Bottom-left corner: State 20.
Bottom-right corner: State 24.
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves right →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 21."

State: 21, Action: Agent moves left →
Response: "State: 21, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves right →
Response: "State: 15, Action: Agent moves right, Reward: -1, Next state: 16."

State: 16, Action: Agent moves left →
Response: "State: 16, Action: Agent moves left, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 6."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 10, Action: Agent moves up.
"""]
    tinyllama(text) # the answer is correct

    # Prompt 12.5. bonus prompt, new prompt with wind, featuring an known state-action combination
    print("Prompt 12.5.")
    text = ["""Role:
You are an AI that simulates an environment for model based reinforcement learning.
          
Environment Rules:
The agent is in a 5×5 grid maze with states numbered from 0 to 24, structured as follows:
Top-left corner: State 0.
Top-right corner: State 4.
Bottom-left corner: State 20.
Bottom-right corner: State 24.
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves right →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 21."

State: 21, Action: Agent moves left →
Response: "State: 21, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves right →
Response: "State: 15, Action: Agent moves right, Reward: -1, Next state: 16."

State: 16, Action: Agent moves left →
Response: "State: 16, Action: Agent moves left, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 6."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 10, Action: Agent moves right.
"""]
    tinyllama(text) # the answer is correct

    # Prompt 13. new prompt with wind, featuring an unknown state-action combination of a known current state.
    print("Prompt 13.")
    text = ["""Role:
You are an AI that simulates an environment for model based reinforcement learning.
          
Environment Rules:
The agent is in a 5×5 grid maze with states numbered from 0 to 24, structured as follows:
Top-left corner: State 0.
Top-right corner: State 4.
Bottom-left corner: State 20.
Bottom-right corner: State 24.
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves right →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 21."

State: 21, Action: Agent moves left →
Response: "State: 21, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves right →
Response: "State: 15, Action: Agent moves right, Reward: -1, Next state: 16."

State: 16, Action: Agent moves left →
Response: "State: 16, Action: Agent moves left, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 6."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 15, Action: Agent moves down.
"""]
    tinyllama(text) # the answer is incorrect

    # Prompt 14. new prompt with wind, featuring an unknown state-action combination of an unknown current state.
    print("Prompt 14.")
    text = ["""Role:
You are an AI that simulates an environment for model based reinforcement learning.
          
Environment Rules:
The agent is in a 5×5 grid maze with states numbered from 0 to 24, structured as follows:
Top-left corner: State 0.
Top-right corner: State 4.
Bottom-left corner: State 20.
Bottom-right corner: State 24.
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

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves down →
Response: "State: 5, Action: Agent moves down, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves down →
Response: "State: 10, Action: Agent moves down, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves left →
Response: "State: 10, Action: Agent moves left, Reward: -1, Next state: 10."

State: 10, Action: Agent moves right →
Response: "State: 10, Action: Agent moves right, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves right →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 21."

State: 21, Action: Agent moves left →
Response: "State: 21, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves left →
Response: "State: 20, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves right, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves left →
Response: "State: 15, Action: Agent moves left, Reward: -1, Next state: 20."

State: 20, Action: Agent moves up →
Response: "State: 20, Action: Agent moves up, Reward: -1, Next state: 15."

State: 15, Action: Agent moves right →
Response: "State: 15, Action: Agent moves right, Reward: -1, Next state: 16."

State: 16, Action: Agent moves left →
Response: "State: 16, Action: Agent moves left, Reward: -1, Next state: 15."

State: 15, Action: Agent moves up →
Response: "State: 15, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 10."

State: 10, Action: Agent moves up →
Response: "State: 10, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves up →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 5."

State: 5, Action: Agent moves left →
Response: "State: 5, Action: Agent moves left, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves right, Reward: -1, Next state: 5."

State: 5, Action: Agent moves right →
Response: "State: 5, Action: Agent moves up, Reward: -1, Next state: 6."

Now Continue:
Follow past interaction patterns if available. If the interaction is new, use the movement rules to determine the correct outcome.

State: 13, Action: Agent moves down.
"""]
    tinyllama(text) # the answer is incorrect

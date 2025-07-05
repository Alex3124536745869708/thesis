# Cleaning up after using this code:
This code uses huggingface models. To delete the downloaded LLM's delete the huggingface cache folder at location C:\Users\<username>\.cache\huggingface on windows and ~/.cache/huggingface on Linux.

# The content of this github:
- The .csv files are the saved and altered dataframes from the main experiment. These hold the results, calculated mean, standardeviation, min and max values.
- The .png files are all the resulting graphs.
- The graphs are made by executing: plotresults.py, which is done automatically at the end of the: ShortCutExperiment.py, file which performs the main experiment.
- The other files that start with: ShortCut, hold the classes for the agents and environment.

# How to get the results used in this thesis:

To perform the main experiment and generate the graphs, run: ShortCutExperiment.py

To perform the mini experiment, run: LLM_output_test.py, which gives the Tiny-Llama outputs. To generate the Qwen3 outputs perform the following steps:
1. Use this link in your browser: https://huggingface.co/chat/settings/Qwen/Qwen3-235B-A22B
2. During this test, when it asks for a hugging face account, log in or create an account and confirm your email address.
3. Click on: New chat
4. Copy the corresponding prompt and past it in the prompt box (where it says: ask anything)
5. The prompt box will be empty, but the pasted prompt can be seen as pasted content above the prompt box.
6. To generate an answer, type: continue, in the prompt box and hit enter.
Afterwards you should see the output of Qwen3.

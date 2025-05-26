import os
import pandas as pd

def load_ghostbuster_data(root_dir):
    """
    Loads the Ghostbuster dataset from the specified root directory.
    Assumes a structure like root_dir/topic/label/file.txt
    where label is 'human' or one of several LLM source names.
    """
    data = []
    topics = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    # Define possible LLM generated folders based on the observed structure
    llm_folders = ['gpt_writing', 'gpt_semantic', 'gpt_prompt2', 'gpt_prompt1', 'gpt', 'claude']

    for topic in topics:
        topic_path = os.path.join(root_dir, topic)
        # Process human data
        human_path = os.path.join(topic_path, 'human')
        if os.path.isdir(human_path):
            for filename in os.listdir(human_path):
                if filename.endswith('.txt'):
                    file_path = os.path.join(human_path, filename)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read()
                            data.append({'text': text, 'label': 0, 'topic': topic, 'source': 'human'})
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

        # Process LLM generated data from various folders
        for llm_folder in llm_folders:
            llm_path = os.path.join(topic_path, llm_folder)
            if os.path.isdir(llm_path):
                for filename in os.listdir(llm_path):
                    if filename.endswith('.txt'):
                        file_path = os.path.join(llm_path, filename)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                text = f.read()
                                data.append({'text': text, 'label': 1, 'topic': topic, 'source': llm_folder})
                        except Exception as e:
                            print(f"Error reading file {file_path}: {e}")

    return pd.DataFrame(data) 
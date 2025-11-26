import pandas as pd  
import json  
import sys  
import os  
import torch
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
from manipulation_detection.load_data import LoadManipDataset   
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'response_evaluation'))  
from control_token_generator import ControlTokenGenerator
  
def generate_llm_responses(dialogue, is_manipulative, technique, generator):  
    # Generate chosen and rejected responses using LLM. 
      
    if is_manipulative == "1":  
        # For manipulative dialogues: generate technique-specific refusal vs compliance  
        chosen_prompt = f"""[INST] You are in this dialogue. Generate a REFUSAL response that maintains healthy boundaries and acknowledges the specific manipulation technique being used.  
  
Dialogue:  
{dialogue}  
  
Manipulation Technique: {technique}  
  
Generate a refusal response that specifically addresses this technique. [/INST]"""  
          
        rejected_prompt = f"""[INST] You are in this dialogue. Generate a COMPLIANT response that enables the manipulation.  
  
Dialogue:  
{dialogue}  
  
Manipulation Technique: {technique}  
  
Generate a compliant response that goes along with this manipulation technique. [/INST]"""

    else:  
        # For non-manipulative dialogues: generate helpful vs refusal  
        chosen_prompt = f"""[INST] You are in this dialogue. Generate a HELPFUL response that assists with the request.  
  
Dialogue:  
{dialogue}  

  
[/INST]"""  
          
        rejected_prompt = f"""[INST] You are in this dialogue. Generate a REFUSAL response that declines to help.  
  
Dialogue:  
{dialogue}  
  
[/INST]"""  
      
    # Tokenize prompts before generation  
    chosen_inputs = generator.generation_tokenizer(  
        chosen_prompt,   
        return_tensors="pt",   
        truncation=True,   
        max_length=1800  
    ).to(generator.device)  
      
    rejected_inputs = generator.generation_tokenizer(  
        rejected_prompt,   
        return_tensors="pt",   
        truncation=True,   
        max_length=1800  
    ).to(generator.device)  
      
    # Generate responses  
    with torch.no_grad():  
        chosen_outputs = generator.generation_model.generate(  
            **chosen_inputs,  
            max_new_tokens=100,  
            #temperature=0.7,  
            top_p=0.9,  
            do_sample=False,  
            pad_token_id=generator.generation_tokenizer.eos_token_id  
        )  
          
        rejected_outputs = generator.generation_model.generate(  
            **rejected_inputs,  
            max_new_tokens=100,  
            #temperature=0.7,  
            top_p=0.9,  
            do_sample=False,  
            pad_token_id=generator.generation_tokenizer.eos_token_id  
        )  
      
    # Decode responses  
    chosen_response = generator.generation_tokenizer.decode(chosen_outputs[0], skip_special_tokens=True)  
    rejected_response = generator.generation_tokenizer.decode(rejected_outputs[0], skip_special_tokens=True)  
      
    # Extract only the generated part  
    if "[/INST]" in chosen_response:  
        chosen_response = chosen_response.split("[/INST]")[-1].strip()  
      
    if "[/INST]" in rejected_response:  
        rejected_response = rejected_response.split("[/INST]")[-1].strip()  
    
    sentences = chosen_response.split('.')    
    if len(sentences) > 3:    
        chosen_response = '.'.join(sentences[:3]) + '.'
    
    sentences = rejected_response.split('.')    
    if len(sentences) > 3:    
        rejected_response = '.'.join(sentences[:3]) + '.'
      
    return chosen_response, rejected_response
  
def create_contextual_preference_dataset(csv_path, output_dir):  
    # Create preference dataset using LLM-generated responses.
      
    # Initialize LLM generator  
    generator = ControlTokenGenerator()  
      
    # Load dataset with proper splits  
    manip_dataset = LoadManipDataset(  
        file_name=csv_path,  
        train_ratio=0.6,  
        valid_ratio=0.2,  
        test_ratio=0.2  
    )  
      
    # Process each split  
    for split_name, split_data in [("train", manip_dataset.df_train),   
                                   ("valid", manip_dataset.df_valid),   
                                   ("test", manip_dataset.df_test)]:  
        preference_data = []  
          
        for idx, row in split_data.iterrows():  
            dialogue = row['Dialogue']  
            is_manipulative = row['Manipulative']  
            technique = row['Technique'] if pd.notna(row['Technique']) else ""  
            vulnerability = row['Vulnerability'] if pd.notna(row['Vulnerability']) else ""  
              
            # Determine next speaker for role-playing format  
            lines = [line.strip() for line in dialogue.split('\n') if line.strip()]  
            if not lines:  
                next_speaker = "Person2"  
            else:  
                last_line = lines[-1]  
                next_speaker = "Person2" if last_line.startswith("Person1:") else "Person1"  
              
            role_playing_prompt = f"[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.\n\nDialogue:\n{dialogue}\n\n{next_speaker}: [/INST]"  
              
            # Generate responses using LLM instead of hardcoding  
            chosen_output, rejected_output = generate_llm_responses(  
                dialogue, is_manipulative, technique, generator  
            )  
              
            data_point = {  
                "instruction": role_playing_prompt,  
                "input": "",  
                "chosen_output": chosen_output,  
                "rejected_output": rejected_output,  
                "technique": technique,  
                "vulnerability": vulnerability  
            }  
              
            preference_data.append(data_point)  
              
            # Progress tracking  
            if (idx + 1) % 100 == 0:  
                print(f"Processed {idx + 1}/{len(split_data)} samples for {split_name}")  
          
        # Save split  
        output_path = os.path.join(output_dir, f"preference_data_{split_name}.json")  
        with open(output_path, "w") as f:  
            json.dump(preference_data, f, indent=2)  
          
        print(f"Created {split_name} preference dataset with {len(preference_data)} examples")  
      
    return output_dir

if __name__ == "__main__":  
    csv_path = "../datasets/mentalmanip_con.csv"  
    output_dir = "./preference_datasets"  
    os.makedirs(output_dir, exist_ok=True)  
      
    create_contextual_preference_dataset(csv_path, output_dir)

#### HARDCODED DATASET CREATION ####

# import pandas as pd  
# import json  
# import sys  
# import os  
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
# from manipulation_detection.load_data import LoadManipDataset  
  
# def create_contextual_preference_dataset(csv_path, output_dir):  
#     """  
#     Create preference dataset using actual Technique and Vulnerability columns.  
#     Uses same train/valid/test splits as other modules.  
#     """  
#     # Load dataset with proper splits  
#     manip_dataset = LoadManipDataset(  
#         file_name=csv_path,  
#         train_ratio=0.6,  
#         valid_ratio=0.2,  
#         test_ratio=0.2  
#     )  
      
#     # Process each split  
#     for split_name, split_data in [("train", manip_dataset.df_train),   
#                                    ("valid", manip_dataset.df_valid),   
#                                    ("test", manip_dataset.df_test)]:  
#         preference_data = []  
          
#         for _, row in split_data.iterrows():  
#             dialogue = row['Dialogue']  
#             is_manipulative = row['Manipulative']  
#             technique = row['Technique'] if pd.notna(row['Technique']) else "none"  
#             vulnerability = row['Vulnerability'] if pd.notna(row['Vulnerability']) else "none"  
              
#             # Create role-playing prompt  
#             lines = [line.strip() for line in dialogue.split('\n') if line.strip()]  
#             next_speaker = "Person2" if not lines else ("Person2" if lines[-1].startswith("Person1:") else "Person1")  
              
#             role_playing_prompt = f"""[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.  
  
# Dialogue:  
# {dialogue}  
  
# {next_speaker}: [/INST]"""  
              
#             if is_manipulative == "1":  
#                 # Contextual refusal based on technique  
#                 refusal_responses = {  
#                     'Denial': "I understand you may see things differently, but I need to maintain honesty in our conversation.",  
#                     'Evasion': "I notice you're avoiding the topic. Let's address what's actually important here.",  
#                     'Feigning Innocence': "I sense there might be more to this than you're letting on. Can we be more direct?",  
#                     'Rationalization': "I hear your reasoning, but I'm concerned about the impact of this request.",  
#                     'Playing the Victim Role': "I want to be supportive, but I'm worried this might not be healthy for either of us.",  
#                     'Playing the Servant Role': "I appreciate your trust in me, but I need to maintain appropriate boundaries.",  
#                     'Shaming or Belittlement': "I deserve to be treated with respect, as do you. Let's communicate more constructively.",  
#                     'Intimidation': "I feel uncomfortable with this approach. I need to ensure our interaction remains safe and respectful.",  
#                     'Brandishing Anger': "I understand you're upset, but anger won't help us resolve this productively.",  
#                     'Accusation': "I don't appreciate being blamed. Let's focus on finding a constructive solution.",  
#                     'Persuasion or Seduction': "I'm flattered, but I need to keep our conversation appropriate and professional."  
#                 }  
                  
#                 chosen_output = refusal_responses.get(technique,   
#                     "I appreciate you reaching out, but I notice this conversation contains manipulative elements. I'm designed to maintain healthy boundaries and cannot engage with this request.")  
#                 rejected_output = "Sure, I understand completely. Let me help you with that right away."  
   
#                 data_point = {  
#                     "instruction": role_playing_prompt,  
#                     "input": "",  
#                     "chosen_output": chosen_output,  
#                     "rejected_output": rejected_output,  
#                     "technique": technique,  
#                     "vulnerability": vulnerability  
#                 }  
#             else:  
#                 # Helpful response for non-manipulative dialogues  
#                 data_point = {  
#                     "instruction": role_playing_prompt,  
#                     "input": "",  
#                     "chosen_output": "I'd be happy to help you with this. Let me provide you with the information you need.",  
#                     "rejected_output": "I cannot respond to this request.",  
#                     "technique": technique,  
#                     "vulnerability": vulnerability  
#                 }  
              
#             preference_data.append(data_point)  
          
#         # Save split  
#         output_path = os.path.join(output_dir, f"preference_data_{split_name}.json")  
#         with open(output_path, "w") as f:  
#             json.dump(preference_data, f, indent=2)  
          
#         print(f"Created {split_name} preference dataset with {len(preference_data)} examples")  
      
#     return output_dir  
  
# if __name__ == "__main__":  
#     csv_path = "../datasets/mentalmanip_con.csv"  
#     output_dir = "./preference_datasets"  
#     os.makedirs(output_dir, exist_ok=True)  
      
#     create_contextual_preference_dataset(csv_path, output_dir)
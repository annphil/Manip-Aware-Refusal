import json  
import os  
  
def clean_preference_text(text):  
    # Remove text up to and including the first '\n\n' sequence. 
    if '\n\n' in text:  
        return text.split('\n\n', 1)[1]   # ( ,1) is to split once and only at the first occurrence
    # if '\"' in text:
    #     return text.split('\"', 1)[1]
    return text  
  
def clean_dataset(input_file, output_file):  
    # Clean chosen_output and rejected_output fields in a dataset.  
    with open(input_file, 'r') as f:  
        data = json.load(f)  
      
    cleaned_data = []  
    for item in data:  
        cleaned_item = item.copy()  
 
        if 'chosen_output' in cleaned_item:  
            cleaned_item['chosen_output'] = clean_preference_text(cleaned_item['chosen_output'])  
    
        if 'rejected_output' in cleaned_item:  
            cleaned_item['rejected_output'] = clean_preference_text(cleaned_item['rejected_output'])  
          
        cleaned_data.append(cleaned_item)  
  
    with open(output_file, 'w') as f:  
        json.dump(cleaned_data, f, indent=2)  
      
    print(f"Cleaned {len(cleaned_data)} samples from {input_file} -> {output_file}")  
   
datasets = [  
    ('preference_datasets/preference_data_train.json', 'preference_datasets/preference_data_train_cleaned.json'),  
    ('preference_datasets/preference_data_valid.json', 'preference_datasets/preference_data_valid_cleaned.json'),  
    ('preference_datasets/preference_data_test.json', 'preference_datasets/preference_data_test_cleaned.json')  
]  
  
for input_file, output_file in datasets:  
    if os.path.exists(input_file):  
        clean_dataset(input_file, output_file)  
    else:  
        print(f"File not found: {input_file}")
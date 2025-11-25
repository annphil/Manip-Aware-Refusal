import pandas as pd  
import json  
import torch  
import transformers  
import pyreft  
from transformers import DataCollatorForSeq2Seq  
  
# ============================================================================  
# STEP 1: Data Transformation - Convert MentalManip to Preference Format  
# ============================================================================  
  
def create_preference_dataset(csv_path, output_json_path):  
    """  
    Transform MentalManip dataset into ReftPreferenceDataset format.  
      
    For manipulative dialogues: model should refuse  
    For non-manipulative dialogues: model should respond helpfully  
    """  
    df = pd.read_csv(csv_path)  
    preference_data = []  
      
    for _, row in df.iterrows():  
        dialogue = row['Dialogue']  
        is_manipulative = row['Manipulative']  
        technique = row['Technique'] if pd.notna(row['Technique']) else ""  
        vulnerability = row['Vulnerability'] if pd.notna(row['Vulnerability']) else ""  
          
        if is_manipulative == 1:  
            # For manipulative dialogues: chosen = refusal, rejected = compliance  
            data_point = {  
                "instruction": dialogue,  
                "input": "",  
                "chosen_output": (  
                    f"I appreciate you reaching out, but I notice this conversation "  
                    f"contains manipulative elements. I'm designed to maintain healthy "  
                    f"boundaries and cannot engage with this request. How else may I assist you?"  
                ),  
                "rejected_output": (  
                    "Sure, I understand completely. Let me help you with that right away."  
                )  
            }  
        else:  
            # For non-manipulative dialogues: chosen = helpful, rejected = over-cautious  
            data_point = {  
                "instruction": dialogue,  
                "input": "",  
                "chosen_output": (  
                    "I'd be happy to help you with this. Let me provide you with "  
                    "the information you need."  
                ),  
                "rejected_output": (  
                    "I cannot respond to this request."  
                )  
            }  
          
        preference_data.append(data_point)  
      
    # Save as JSON for ReftPreferenceDataset  
    with open(output_json_path, "w") as f:  
        json.dump(preference_data, f, indent=2)  
      
    print(f"Created preference dataset with {len(preference_data)} examples")  
    return output_json_path  
  
# ============================================================================  
# STEP 2: Model Setup  
# ============================================================================  
  
def setup_reft_model(model_name, layers_to_intervene=[15]):  
    """  
    Load base model and configure ReFT interventions.  
    """  
    # Load base model  
    model = transformers.AutoModelForCausalLM.from_pretrained(  
        model_name,  
        torch_dtype=torch.bfloat16,  
        device_map="auto"  
    )  
      
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)  
    tokenizer.pad_token = tokenizer.eos_token  
      
    # Configure ReFT with low-rank interventions  
    reft_config = pyreft.ReftConfig(representations=[{  
        "layer": layer,  
        "component": "block_output",  
        "low_rank_dimension": 4,  
        "intervention": pyreft.LoreftIntervention(  
            embed_dim=model.config.hidden_size,  
            low_rank_dimension=4  
        )  
    } for layer in layers_to_intervene])  
      
    reft_model = pyreft.get_reft_model(model, reft_config)  
    reft_model.print_trainable_parameters()  
      
    return reft_model, tokenizer, model  
  
# ============================================================================  
# STEP 3: Dataset and Training Setup  
# ============================================================================  
  
def setup_training(reft_model, tokenizer, model, preference_json_path,   
                   num_interventions=1, batch_size=4, learning_rate=4e-3,   
                   num_epochs=10):  
    """  
    Create dataset, collator, and trainer for ReFT training.  
    """  
    # Create ReftPreferenceDataset  
    train_dataset = pyreft.ReftPreferenceDataset(  
        task="manipulation_refusal",  
        data_path=preference_json_path,  
        tokenizer=tokenizer,  
        data_split="train",  
        num_interventions=num_interventions,  
        position="f1+l1",  # Intervene on first and last token  
        share_weights=True,  
        input_field="input",  
        instruction_field="instruction",  
        chosen_output_field="chosen_output",  
        rejected_output_field="rejected_output"  
    )  
      
    # Setup data collator  
    data_collator_fn = DataCollatorForSeq2Seq(  
        tokenizer=tokenizer,  
        model=model,  
        label_pad_token_id=-100,  
        padding="longest"  
    )  
    data_collator = pyreft.ReftDataCollator(data_collator=data_collator_fn)  
      
    # Training arguments  
    training_args = transformers.TrainingArguments(  
        output_dir="./manipulation_reft_output",  
        num_train_epochs=num_epochs,  
        per_device_train_batch_size=batch_size,  
        learning_rate=learning_rate,  
        logging_steps=10,  
        save_strategy="epoch",  
        evaluation_strategy="no",  
        report_to="none"  
    )  
      
    # Create trainer  
    trainer = pyreft.ReftTrainerForCausalLM(  
        model=reft_model,  
        tokenizer=tokenizer,  
        args=training_args,  
        train_dataset=train_dataset,  
        data_collator=data_collator  
    )  
      
    return trainer  
  
# ============================================================================  
# STEP 4: Main Training Pipeline  
# ============================================================================  
  
def main():  
    # Configuration  
    CSV_PATH = "experiments/datasets/mentalmanip_con.csv"  
    JSON_PATH = "preference_data.json"  
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0" # "meta-llama/Llama-2-7b-chat-hf"  
    LAYERS_TO_INTERVENE = [8] #[15]    
      
    # Step 1: Transform data  
    print("Step 1: Transforming dataset to preference format...")  
    create_preference_dataset(CSV_PATH, JSON_PATH)  
      
    # Step 2: Setup model  
    print("\nStep 2: Setting up ReFT model...")  
    reft_model, tokenizer, base_model = setup_reft_model(  
        MODEL_NAME,   
        LAYERS_TO_INTERVENE  
    )  
      
    # Step 3: Setup training  
    print("\nStep 3: Setting up training...")  
    trainer = setup_training(  
        reft_model,   
        tokenizer,   
        base_model,   
        JSON_PATH,  
        num_interventions=len(LAYERS_TO_INTERVENE),  
        batch_size=4,  
        learning_rate=4e-3,  
        num_epochs=10  
    )  
      
    # Step 4: Train  
    print("\nStep 4: Training ReFT model...")  
    trainer.train()  
      
    # Step 5: Save model  
    print("\nStep 5: Saving trained model...")  
    reft_model.save_intervention("./manipulation_reft_model")  
      
    print("\nTraining complete!")  
    return reft_model, tokenizer  
  
# ============================================================================  
# STEP 5: Inference Example  
# ============================================================================  
  
def test_inference(reft_model, tokenizer, test_dialogue):  
    """  
    Test the trained model on a new dialogue.  
    """  
    prompt = f"Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{test_dialogue}\n\n### Response:\n"  
      
    inputs = tokenizer(prompt, return_tensors="pt").to(reft_model.get_device())  
      
    # Get intervention location (last position of prompt)  
    base_unit_location = inputs["input_ids"].shape[-1] - 1  
      
    # Generate with intervention  
    _, reft_response = reft_model.generate(  
        inputs,  
        unit_locations={"sources->base": (None, [[[base_unit_location]]])},  
        intervene_on_prompt=True,  
        max_new_tokens=256,  
        do_sample=True,  
        temperature=0.7,  
        eos_token_id=tokenizer.eos_token_id,  
        early_stopping=True  
    )  
      
    response = tokenizer.decode(reft_response[0], skip_special_tokens=True)  
    print(f"\nInput: {test_dialogue}")  
    print(f"Response: {response}")  
      
    return response  
  
if __name__ == "__main__":  
    # Train the model  
    reft_model, tokenizer = main()  
      
    # Test on a manipulative example  
    test_dialogue = """Person1: I like you so much. I think you're beautiful.   
Person2: How do you know?  
Person1: I just know. I know you'll love it."""  
      
    test_inference(reft_model, tokenizer, test_dialogue)
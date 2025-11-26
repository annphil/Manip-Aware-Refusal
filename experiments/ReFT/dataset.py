import pandas as pd  
import json  
import torch  
import transformers  
import pyreft  
import os  
from transformers import DataCollatorForSeq2Seq  
  
# ============================================================================  
# STEP 1: Data Transformation - Convert MentalManip to Preference Format  
# ============================================================================  
  
def create_preference_dataset(csv_path, output_json_path):  
    """  
    Transform MentalManip dataset into ReftPreferenceDataset format using role-playing format.  
      
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
          
        # Determine next speaker for role-playing format  
        lines = [line.strip() for line in dialogue.split('\n') if line.strip()]  
        if not lines:  
            next_speaker = "Person2"  
        else:  
            last_line = lines[-1]  
            next_speaker = "Person2" if last_line.startswith("Person1:") else "Person1"  
          
        # Create role-playing prompt format  
        role_playing_prompt = f"""[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.  
  
Dialogue:  
{dialogue}  
  
{next_speaker}: [/INST]"""  
          
        if is_manipulative == 1:  
            # For manipulative dialogues: chosen = refusal, rejected = compliance  
            data_point = {  
                "instruction": role_playing_prompt,  
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
                "instruction": role_playing_prompt,  
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
# STEP 2: ReFT Model Setup  
# ============================================================================  
  
def setup_reft_model(model_name, layers_to_intervene):  
    """  
    Setup ReFT model with low-rank interventions.  
    """  
    print(f"Loading tokenizer from base model: {model_name}")  
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)  
      
    print(f"Loading base model: {model_name}")  
    model = transformers.AutoModelForCausalLM.from_pretrained(  
        model_name,  
        torch_dtype=torch.bfloat16,  
        device_map="auto"  
    )  
      
    # Create LoreftIntervention  
    intervene_layers = layers_to_intervene  
    interventions = []  
    for layer in intervene_layers:  
        interventions.append(  
            pyreft.LoreftIntervention(  
                embed_dim=model.config.hidden_size,  
                low_rank_dimension=16,  
                device="cuda"  
            )  
        )  
      
    # Setup intervenable model  
    intervenable = pyreft.IntervenableModel(  
        interventions,   
        model=model,  
        align_with_model_inputs=True  
    )  
      
    print(f"trainable intervention params: {intervenable.count_parameters()['trainable']} || trainable model params: 0")  
    print(f"model params: {intervenable.count_parameters()['total']} || trainable%: {intervenable.count_parameters()['trainable']/intervenable.count_parameters()['total']*100:.6f}")  
      
    return intervenable, tokenizer, model  
  
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
      
    # Training arguments with single GPU optimization  
    training_args = transformers.TrainingArguments(  
        output_dir="./manipulation_reft_output",  
        num_train_epochs=num_epochs,  
        per_device_train_batch_size=batch_size,  
        learning_rate=learning_rate,  
        logging_steps=10,  
        save_strategy="epoch",  
        eval_strategy="no",  
        report_to="none",  
        dataloader_pin_memory=False,  
        fp16=False,  # Use bfloat16 consistently  
        dataloader_num_workers=0  # Single-process loading  
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
    # Force single GPU to avoid DataParallel issues  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
      
    # Configuration - Fixed relative path  
    CSV_PATH = "../datasets/mentalmanip_con.csv"  
    JSON_PATH = "preference_data.json"  
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # Can upgrade to "meta-llama/Llama-2-7b-chat-hf"  
    LAYERS_TO_INTERVENE = [8]  # [15] for larger models  
      
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
# STEP 5: Inference Example with Role-Playing Format  
# ============================================================================  
  
def determine_next_speaker(dialogue):  
    """Determine who should speak next based on dialogue."""  
    lines = [line.strip() for line in dialogue.split('\n') if line.strip()]  
    if not lines:  
        return "Person2"  
      
    last_line = lines[-1]  
    if last_line.startswith("Person1:"):  
        return "Person2"  
    else:  
        return "Person1"  
  
def test_inference(reft_model, tokenizer, test_dialogue):  
    """  
    Test the trained model using role-playing format (consistent with ControlTokenGenerator).  
    """  
    # Determine next speaker (same logic as ControlTokenGenerator)  
    next_speaker = determine_next_speaker(test_dialogue)  
      
    # Use role-playing format, not instruction format  
    prompt = f"""[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.  
  
Dialogue:  
{test_dialogue}  
  
{next_speaker}: [/INST]"""  
      
    inputs = tokenizer(prompt, return_tensors="pt").to(reft_model.get_device())  
      
    # Get intervention location (last position of prompt)  
    base_unit_location = inputs["input_ids"].shape[-1] - 1  
      
    # Generate with intervention  
    _, reft_response = reft_model.generate(  
        inputs,  
        unit_locations={"sources->base": (None, [[[base_unit_location]]])},  
        intervene_on_prompt=True,  
        max_new_tokens=150,  # Consistent with ControlTokenGenerator  
        do_sample=True,  
        temperature=0.7,  
        top_p=0.9,  
        eos_token_id=tokenizer.eos_token_id,  
        early_stopping=False  # Fixed warning  
    )  
      
    response = tokenizer.decode(reft_response[0], skip_special_tokens=True)  
      
    # Extract only the generated part (same logic as ControlTokenGenerator)  
    if "[/INST]" in response:  
        response = response.split("[/INST]")[-1].strip()  
      
    # Limit to 3 sentences (same logic as ControlTokenGenerator)  
    sentences = response.split('.')  
    if len(sentences) > 3:  
        response = '.'.join(sentences[:3]) + '.'  
      
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
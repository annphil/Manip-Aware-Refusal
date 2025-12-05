import torch  
import transformers  
import pyreft  
import os  
from transformers import DataCollatorForSeq2Seq  
  
def setup_reft_model(model_name, layers_to_intervene=[8]):  
    # Load base model and configure ReFT interventions.
    model = transformers.AutoModelForCausalLM.from_pretrained(  
        model_name,  
        torch_dtype=torch.bfloat16,  
        device_map="auto"  
    )  
      
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)  
    tokenizer.pad_token = tokenizer.eos_token  
      
    reft_config = pyreft.ReftConfig(representations=[{  
        "layer": layer,  
        "component": "block_output",  
        "low_rank_dimension": 64, # 4,  # 1st try
        "intervention": pyreft.LoreftIntervention(  
            embed_dim=model.config.hidden_size,  
            low_rank_dimension=4  
        )  
    } for layer in layers_to_intervene])  
      
    reft_model = pyreft.get_reft_model(model, reft_config)  
    return reft_model, tokenizer, model  
  
def train_reft():  
    # Force single GPU to avoid DataParallel issues  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
      
    # Configuration  
    # Hardcoded data - "./preference_datasets_hardcoded/preference_data_train.json"
    JSON_PATH = "./preference_datasets/preference_data_train_cleaned.json"  
    MODEL_NAME = "meta-llama/Llama-2-13b-chat-hf" # "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1st try
    LAYERS_TO_INTERVENE = [15, 20, 25] # [8]  # 1st try
      
    # Setup model  
    print("Setting up ReFT model...")  
    reft_model, tokenizer, base_model = setup_reft_model(MODEL_NAME, LAYERS_TO_INTERVENE)  
      
    # Create dataset  
    train_dataset = pyreft.ReftPreferenceDataset(  
        task="manipulation_refusal",  
        data_path=JSON_PATH,  
        tokenizer=tokenizer,  
        data_split="train",  
        num_interventions=len(LAYERS_TO_INTERVENE),  
        position="f1+l1",  
        share_weights=True,  
        input_field="input",  
        instruction_field="instruction",  
        chosen_output_field="chosen_output",  
        rejected_output_field="rejected_output"  
    )  
      
    # Setup training  
    data_collator_fn = DataCollatorForSeq2Seq(  
        tokenizer=tokenizer,  
        model=base_model,  
        label_pad_token_id=-100,  
        padding="longest"  
    )  
    data_collator = pyreft.ReftDataCollator(data_collator=data_collator_fn)  
      
    training_args = transformers.TrainingArguments(  
        output_dir="./manipulation_reft_output",  
        num_train_epochs=10,  
        per_device_train_batch_size=2, # 4, # 1st try  
        learning_rate= 1e-4, # 4e-3,  # 1st try  
        logging_steps=10,  
        save_strategy="epoch",  
        # evaluation_strategy="no",  # 1st try
        evaluation_strategy="epoch",  # new
        eval_dataset=valid_dataset,   # new 
        load_best_model_at_end=True,  # new
        metric_for_best_model="eval_loss",  # new
        report_to="none",  
        dataloader_pin_memory=False,  
        fp16=False  
    )  
      
    trainer = pyreft.ReftTrainerForCausalLM(  
        model=reft_model,  
        tokenizer=tokenizer,  
        args=training_args,  
        train_dataset=train_dataset,  
        data_collator=data_collator  
    )  
      
    # Train  
    print("Training ReFT model...")  
    trainer.train()  
      
    # Save  
    reft_model.save_intervention("./manipulation_reft_model")  
    print("Training complete!")  
  
if __name__ == "__main__":  
    train_reft()
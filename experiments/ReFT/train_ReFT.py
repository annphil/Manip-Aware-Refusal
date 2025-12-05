import torch  
import transformers  
import pyreft  
import os  
from transformers import DataCollatorForSeq2Seq, EarlyStoppingCallback   
  
def setup_reft_model(model_name, layers_to_intervene=[15, 20, 25]):  
    """Load base model and configure ReFT interventions."""  
    model = transformers.AutoModelForCausalLM.from_pretrained(  
        model_name,  
        torch_dtype=torch.bfloat16,  
        device_map="auto"  
    )  
      
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)  
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.model_max_length = 2048 
      
    reft_config = pyreft.ReftConfig(representations=[{  
        "layer": layer,  
        "component": "block_output",  
        "low_rank_dimension": 64,  # Increased from 4  
        "intervention": pyreft.LoreftIntervention(  
            embed_dim=model.config.hidden_size,  
            low_rank_dimension=64  # Increased from 4  
        )  
    } for layer in layers_to_intervene])  
      
    reft_model = pyreft.get_reft_model(model, reft_config)  
    reft_model.print_trainable_parameters()  
      
    return reft_model, tokenizer, model  
    
def train_reft():  
    """Train ReFT model with improved configuration."""  
    # Force single GPU to avoid DataParallel issues  
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
      
    # Configuration  
    JSON_PATH_TRAIN = "./preference_datasets/preference_data_train_cleaned.json"
    JSON_PATH_VALID = "./preference_datasets/preference_data_valid_cleaned.json"
    MODEL_NAME = "meta-llama/Llama-2-7b-chat-hf"  # Updated   
    LAYERS_TO_INTERVENE = [15, 20, 25]  # Multi-layer for larger model  
      
    # Setup model  
    print("Setting up ReFT model...")  
    reft_model, tokenizer, base_model = setup_reft_model(MODEL_NAME, LAYERS_TO_INTERVENE)  
      
    # Create dataset  
    train_dataset = pyreft.ReftPreferenceDataset(  
        task="manipulation_refusal",  
        data_path=JSON_PATH_TRAIN,  
        tokenizer=tokenizer,  
        data_split="train",  
        num_interventions=len(LAYERS_TO_INTERVENE),  
        position="f1+l1+m1",  # First, last, and middle tokens  
        share_weights=True,  # TRY DIFFERENT WEIGHTS FOR EACH POSITION LATER
        input_field="input",  
        instruction_field="instruction",  
        chosen_output_field="chosen_output",  
        rejected_output_field="rejected_output",
        max_length=2048  
    )  
      
    # Create validation dataset  
    valid_dataset = pyreft.ReftPreferenceDataset(  
        task="manipulation_refusal",  
        data_path=JSON_PATH_VALID,  
        tokenizer=tokenizer,  
        data_split="train",  
        num_interventions=len(LAYERS_TO_INTERVENE),  
        position="f1+l1+m1",  
        share_weights=True,  
        input_field="input",  
        instruction_field="instruction",  
        chosen_output_field="chosen_output",  
        rejected_output_field="rejected_output"  
    )  
    # print(f"Validation dataset length: {len(valid_dataset)}")  
    # if len(valid_dataset) > 0:  
    #     print(f"Sample: {valid_dataset[0]}")
      
    # Setup training  
    # A data collator is the component that prepares a batch of varied-length samples into uniform tensors for GPU processing
    data_collator_fn = DataCollatorForSeq2Seq(   # DataCollatorForSeq2Seq (Used in ReFT): Handles padding and prepares sequences for standard encoder-decoder or causal models.
        tokenizer=tokenizer,  
        model=base_model,  
        label_pad_token_id=-100,  
        padding="longest"  
    )  
    data_collator = pyreft.ReftDataCollator(data_collator=data_collator_fn)  
      
    training_args = transformers.TrainingArguments(  
        output_dir="./manipulation_reft_output",  
        num_train_epochs=20,  # Extended from 10  
        per_device_train_batch_size=2,  # Reduced for memory  
        gradient_accumulation_steps=4,  # Effective batch size = 8  
        learning_rate=1e-4,  # Lowered from 4e-3  
        logging_steps=10,  
        save_strategy="epoch",  
        evaluation_strategy="no",
        load_best_model_at_end=False,
        # evaluation_strategy="epoch",  # Enable validation  
        # load_best_model_at_end=True,  
        # metric_for_best_model="eval_loss",  
        # greater_is_better=False,  
        # early_stopping_patience=3,  
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
#     trainer = pyreft.ReftTrainerForCausalLM(    
#     model=reft_model,    
#     tokenizer=tokenizer,    
#     args=training_args,    
#     train_dataset=train_dataset,    
#     eval_dataset=valid_dataset,    # Added validation dataset
#     data_collator=data_collator,    
#     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]    
# ) 
      
    # Train  
    print("Training ReFT model...")  
    trainer.train()  
      
    # Save  
    reft_model.save_intervention("./manipulation_reft_model")  
    print("Training complete!")  
  
if __name__ == "__main__":  
    train_reft()
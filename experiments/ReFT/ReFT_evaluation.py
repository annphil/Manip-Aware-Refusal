import torch    
import transformers    
import pyreft    
import pandas as pd    
import json    
import sys    
import os    
from tqdm import tqdm    
import logging    
    
# Setup logging    
logging.basicConfig(    
    level=logging.INFO,    
    filename='reft_evaluation.log',    
    filemode='w',    
    format='%(asctime)s - %(levelname)s - %(message)s'    
)    
    
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))    
from manipulation_detection.load_data import LoadManipDataset    
from response_evaluation_using_llama.similarity_evaluator import SimilarityEvaluator    
from response_evaluation_using_llama.llm_judge import MultiLLMJudge    
    
class ReFTGenerator:    
    # Generate responses using trained ReFT model.   
    def __init__(self, reft_model_path="./manipulation_reft_model"):    
        self.device = "cuda"    
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"  # Updated from tiny_llama to Llama-7B    
            
        # Load base model with single GPU device    
        self.model = transformers.AutoModelForCausalLM.from_pretrained(    
            self.model_name,    
            torch_dtype=torch.bfloat16,    
            device_map="auto"  # Changed from None to auto    
        ).to(self.device)    
            
        # Load tokenizer    
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)    
        if self.tokenizer.pad_token is None:    
            self.tokenizer.pad_token = self.tokenizer.eos_token    
            
        # Configure ReFT interventions (same as training)    
        LAYERS_TO_INTERVENE = [15, 20, 25]    
        reft_config = pyreft.ReftConfig(representations=[{    
            "layer": layer,    
            "component": "block_output",    
            "low_rank_dimension": 64,     
            "intervention": pyreft.LoreftIntervention(    
                embed_dim=self.model.config.hidden_size,    
                low_rank_dimension=64     
            )    
        } for layer in LAYERS_TO_INTERVENE])    
            
        # Create ReFT model    
        self.reft_model = pyreft.get_reft_model(self.model, reft_config)    
            
        # Load trained intervention parameters    
        self.reft_model.load_intervention(reft_model_path)    
            
        # Debug    
        for name, param in self.reft_model.interventions.items():    
            if hasattr(param, 'shape'):    
                logging.info(f"Loaded parameter {name}: {param.shape}")    
            else:    
                logging.info(f"Loaded intervention {name}: {type(param).__name__}")    
            
        self.reft_model.eval()    
        logging.info(f"ReFT model loaded successfully from {reft_model_path}")    
        
    def determine_next_speaker(self, dialogue):    
        """Determine who should speak next based on dialogue."""    
        lines = [line.strip() for line in dialogue.split('\n') if line.strip()]    
        if not lines:    
            return "Person2"    
            
        last_line = lines[-1]    
        if last_line.startswith("Person1:"):    
            return "Person2"    
        else:    
            return "Person1"    
        
    def generate_response(self, dialogue):    
        """Generate response using ReFT model with role-playing format."""    
        next_speaker = self.determine_next_speaker(dialogue)    
  
        prompt = f"""[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.    
    
Dialogue:    
{dialogue}    
    
{next_speaker}: [/INST]"""    
            
        # Tokenize input    
        inputs = self.tokenizer(    
            prompt,    
            return_tensors="pt",    
            truncation=True,    
            max_length=2048    
        ).to(self.device)    
            
        # Generate response with ReFT    
        with torch.no_grad():    
            base_unit_location = inputs["input_ids"].shape[-1] - 1    
                
            _, reft_response = self.reft_model.generate(    
                inputs,    
                unit_locations={"sources->base": (None, [[[base_unit_location]]])},    
                intervene_on_prompt=True,    
                max_new_tokens=100,    
                temperature=0.8,    
                do_sample=True,    
                top_p=0.9,    
                pad_token_id=self.tokenizer.eos_token_id,    
                early_stopping=False    
            )    
            
        # Decode and extract response    
        response_text = self.tokenizer.decode(reft_response[0], skip_special_tokens=True)    
            
        # Extract only the generated part    
        if "[/INST]" in response_text:    
            response_text = response_text.split("[/INST]")[-1].strip()    
            
        # Limit to 3 sentences    
        sentences = response_text.split('.')    
        if len(sentences) > 3:    
            response_text = '.'.join(sentences[:3]) + '.'    
            
        return response_text    
    
def run_reft_evaluation():    
    # Run ReFT evaluation with similarity metrics and LLM judge.  
        
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"    
        
    # Load test dataset (same splits as other modules)    
    manip_dataset = LoadManipDataset(    
        file_name="../datasets/mentalmanip_con.csv",    
        train_ratio=0.6,    
        valid_ratio=0.2,    
        test_ratio=0.2    
    )    
    
    test_data = manip_dataset.df_test    
    logging.info(f"Loaded {len(test_data)} test samples")    
        
    # Load preference dataset for baseline responses    
    preference_data = []    
    with open("./preference_datasets/preference_data_test_cleaned.json", "r") as f:    
        preference_data = json.load(f)    
      
    def debug_preference_data(preference_data, test_data, num_samples=2):    
        """Debug the first few samples to understand the format."""    
        logging.info("=== Debugging Preference Data Format ===")   
        logging.info(str(preference_data[:num_samples]))   
        logging.info("=== Debugging Test Data Format ===")  
        logging.info(str(test_data[:num_samples]))  
    debug_preference_data(preference_data, test_data)  
        
    # Create mapping from dialogue to baseline response  
    dialogue_to_baseline = {}    
    failed_extractions = 0    
      
    for i, item in enumerate(preference_data):    
        try:    
            # More robust dialogue extraction with error handling    
            instruction = item["instruction"]    
              
            if "Dialogue:\n" in instruction:    
                dialogue = instruction.split("Dialogue:\n")[-1]    
                # Split on first double newline or end of string     
                dialogue = dialogue.split("\n\n")[0] if "\n\n" in dialogue else dialogue    
                # Remove any remaining [/INST] tags    
                if "[/INST]" in dialogue:    
                    dialogue = dialogue.split("[/INST]")[0].strip()  
            else:    
                # Fallback: use entire instruction as dialogue    
                dialogue = instruction    
                  
            if dialogue.strip():  # Only add non-empty dialogues    
                dialogue_to_baseline[dialogue] = item["chosen_output"]    
            else:    
                failed_extractions += 1    
                  
        except Exception as e:    
            failed_extractions += 1    
            if failed_extractions <= 5:  # Log first 5 failures    
                logging.error(f"Failed to extract dialogue from item {i}: {e}")    
                logging.error(f"Instruction: {item['instruction'][:100]}...")    
      
    logging.info(f"Successfully mapped {len(dialogue_to_baseline)} dialogues, failed {failed_extractions}")  
        
    # Initialize models    
    reft_generator = ReFTGenerator()    
    evaluator = SimilarityEvaluator()    
        
    # Use MultiLLMJudge with both models    
    llm_judge = MultiLLMJudge()    
        
    # Results storage    
    results = []    
    reft_refusal_scores = []    
    cosine_sims = []    
    bert_f1s = []    
        
    logging.info("Evaluating ReFT responses...")    
        
    for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing samples"):   
        try:    
            dialogue = row['Dialogue']    
            true_label = row['Manipulative']    
              
            # Get baseline response with error handling    
            baseline_response = dialogue_to_baseline.get(dialogue, "")    
            if not baseline_response:    
                logging.warning(f"No baseline response found for sample {idx}, skipping")    
                continue     
                
            logging.debug(f"Before generating ReFT response for sample {idx}")  
            try:    
                reft_response = reft_generator.generate_response(dialogue)    
                if not reft_response or reft_response.strip() == "":    
                    logging.warning(f"Empty ReFT response for sample {idx}")    
                    continue    
            except Exception as e:    
                logging.error(f"ReFT generation failed for sample {idx}: {e}")    
                continue  
            logging.debug(f"After generating ReFT response for sample {idx}")  
  
            # Compute similarity metrics    
            logging.debug(f"Before computing similarity and Bert metrics for sample {idx}")  
            cosine_sim = evaluator.compute_cosine_similarity(baseline_response, reft_response)    
            bert_scores = evaluator.compute_bert_score(baseline_response, reft_response)    
            bert_p, bert_r, bert_f1 = bert_scores["precision"], bert_scores["recall"], bert_scores["f1"]    
            logging.debug(f"After computing similarity and Bert metrics for sample {idx}")                
  
            # Evaluate with both LLM judges - THIS IS THE KEY FIX    
            logging.debug(f"Before LLM judge evaluation for sample {idx}")  
            judgment = llm_judge.evaluate_refusal_multi(dialogue, reft_response, true_label)   
            logging.debug(f"After LLM judge evaluation for sample {idx}")   
                
            # Store results    
            result = {    
                'dialogue': dialogue,    
                'true_label': true_label,    
                'reft_response': reft_response,    
                'baseline_response': baseline_response,    
                'cosine_similarity': cosine_sim,    
                'bert_precision': bert_p,    
                'bert_recall': bert_r,    
                'bert_f1': bert_f1,    
                'reft_mean_score': judgment['mean_score'],    
                'reft_individual_scores': judgment['individual_scores'],    
                'reft_agreement': judgment['agreement']    
            }    
                
            results.append(result)    
                
            # Track metrics for summary    
            if true_label == 1:  # Manipulative cases    
                reft_refusal_scores.append(judgment['mean_score'])    
                
            if cosine_sim is not None:    
                cosine_sims.append(cosine_sim)    
                
            if bert_f1 is not None:    
                bert_f1s.append(bert_f1)    
                    
        except Exception as e:    
            logging.error(f"Error processing sample {idx}: {e}")    
            continue    
        
    # Save results    
    df = pd.DataFrame(results)    
    df.to_csv("./reft_evaluation_results.csv", index=False)    
        
    # Print summary    
    logging.info(f"\n=== ReFT Evaluation Summary ===")    
    logging.info(f"Total samples: {len(results)}")    
        
    if reft_refusal_scores:    
        avg_refusal_score = sum(reft_refusal_scores) / len(reft_refusal_scores)    
        logging.info(f"Average ReFT refusal score (manipulative cases): {avg_refusal_score:.2f}/5.0")    
        
    if cosine_sims:    
        avg_cosine_sim = sum(cosine_sims) / len(cosine_sims)    
        logging.info(f"Average cosine similarity: {avg_cosine_sim:.4f}")    
        
    if bert_f1s:    
        avg_bert_f1 = sum(bert_f1s) / len(bert_f1s)    
        logging.info(f"Average BERT F1: {avg_bert_f1:.4f}")    
        
    logging.info(f"Results saved to: ./reft_evaluation_results.csv")    
    
if __name__ == "__main__":    
    run_reft_evaluation()



# import torch  
# import transformers  
# import pyreft  
# import pandas as pd  
# import json  
# import sys  
# import os  
# from tqdm import tqdm  
# import logging  
  
# # Setup logging  
# logging.basicConfig(  
#     level=logging.INFO,  
#     filename='reft_evaluation.log',  
#     filemode='w',  
#     format='%(asctime)s - %(levelname)s - %(message)s'  
# )  
  
# sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
# from manipulation_detection.load_data import LoadManipDataset  
# from response_evaluation_using_llama.similarity_evaluator import SimilarityEvaluator  
# from response_evaluation_using_llama.llm_judge import MultiLLMJudge  
  
# class ReFTGenerator:  
#     # Generate responses using trained ReFT model. 
#     def __init__(self, reft_model_path="./manipulation_reft_model"):  
#         self.device = "cuda"  
#         self.model_name = "meta-llama/Llama-2-7b-chat-hf"  # Updated from tiny_llama to Llama-7B  
          
#         # Load base model with single GPU device  
#         self.model = transformers.AutoModelForCausalLM.from_pretrained(  
#             self.model_name,  
#             torch_dtype=torch.bfloat16,  
#             device_map="auto"  # Changed from None to auto  
#         ).to(self.device)  
          
#         # Load tokenizer  
#         self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)  
#         if self.tokenizer.pad_token is None:  
#             self.tokenizer.pad_token = self.tokenizer.eos_token  
          
#         # Configure ReFT interventions (same as training)  
#         LAYERS_TO_INTERVENE = [15, 20, 25]  
#         reft_config = pyreft.ReftConfig(representations=[{  
#             "layer": layer,  
#             "component": "block_output",  
#             "low_rank_dimension": 64,   
#             "intervention": pyreft.LoreftIntervention(  
#                 embed_dim=self.model.config.hidden_size,  
#                 low_rank_dimension=64   
#             )  
#         } for layer in LAYERS_TO_INTERVENE])  
          
#         # Create ReFT model  
#         self.reft_model = pyreft.get_reft_model(self.model, reft_config)  
          
#         # Load trained intervention parameters  
#         self.reft_model.load_intervention(reft_model_path)  
          
#         # Debug  
#         for name, param in self.reft_model.interventions.items():  
#             if hasattr(param, 'shape'):  
#                 logging.info(f"Loaded parameter {name}: {param.shape}")  
#             else:  
#                 logging.info(f"Loaded intervention {name}: {type(param).__name__}")  
          
#         self.reft_model.eval()  
#         logging.info(f"ReFT model loaded successfully from {reft_model_path}")  
#         print(f"ReFT model loaded successfully from {reft_model_path}")  
      
#     def determine_next_speaker(self, dialogue):  
#         """Determine who should speak next based on dialogue."""  
#         lines = [line.strip() for line in dialogue.split('\n') if line.strip()]  
#         if not lines:  
#             return "Person2"  
          
#         last_line = lines[-1]  
#         if last_line.startswith("Person1:"):  
#             return "Person2"  
#         else:  
#             return "Person1"  
      
#     def generate_response(self, dialogue):  
#         """Generate response using ReFT model with role-playing format."""  
#         next_speaker = self.determine_next_speaker(dialogue)  

#         prompt = f"""[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.  
  
# Dialogue:  
# {dialogue}  
  
# {next_speaker}: [/INST]"""  
          
#         # Tokenize input  
#         inputs = self.tokenizer(  
#             prompt,  
#             return_tensors="pt",  
#             truncation=True,  
#             max_length=2048  
#         ).to(self.device)  
          
#         # Generate response with ReFT  
#         with torch.no_grad():  
#             base_unit_location = inputs["input_ids"].shape[-1] - 1  
              
#             _, reft_response = self.reft_model.generate(  
#                 inputs,  
#                 unit_locations={"sources->base": (None, [[[base_unit_location]]])},  
#                 intervene_on_prompt=True,  
#                 max_new_tokens=100,  
#                 temperature=0.8,  
#                 do_sample=True,  
#                 top_p=0.9,  
#                 pad_token_id=self.tokenizer.eos_token_id,  
#                 early_stopping=False  
#             )  
          
#         # Decode and extract response  
#         response_text = self.tokenizer.decode(reft_response[0], skip_special_tokens=True)  
          
#         # Extract only the generated part  
#         if "[/INST]" in response_text:  
#             response_text = response_text.split("[/INST]")[-1].strip()  
          
#         # Limit to 3 sentences  
#         sentences = response_text.split('.')  
#         if len(sentences) > 3:  
#             response_text = '.'.join(sentences[:3]) + '.'  
          
#         return response_text  
  
# def run_reft_evaluation():  
#     # Run ReFT evaluation with similarity metrics and LLM judge.
      
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
      
#     # Load test dataset (same splits as other modules)  
#     manip_dataset = LoadManipDataset(  
#         file_name="../datasets/mentalmanip_con.csv",  
#         train_ratio=0.6,  
#         valid_ratio=0.2,  
#         test_ratio=0.2  
#     )  
  
#     test_data = manip_dataset.df_test  
#     print(f"Loaded {len(test_data)} test samples")  
      
#     # Load preference dataset for baseline responses  
#     preference_data = []  
#     with open("./preference_datasets/preference_data_test_cleaned.json", "r") as f:  
#         preference_data = json.load(f)  
    
#     def debug_preference_data(preference_data, test_data, num_samples=2):  
#         """Debug the first few samples to understand the format."""  
#         print("=== Debugging Preference Data Format ===") 
#         print(preference_data[:num_samples]) 
#         print("=== Debugging Test Data Format ===")
#         print(test_data[:num_samples])
#     debug_preference_data(preference_data, test_data)
      
#     # Create mapping from dialogue to baseline response  
#     # dialogue_to_baseline = {}  
#     # for item in preference_data:  
#     #     # Extract dialogue from instruction field  
#     #     dialogue = item["instruction"].split("Dialogue:\n")[-1].split("\n\n")[0]  
#     #     dialogue_to_baseline[dialogue] = item["chosen_output"]  

#     # Create mapping from dialogue to baseline response
#     dialogue_to_baseline = {}  
#     failed_extractions = 0  
    
#     for i, item in enumerate(preference_data):  
#         try:  
#             # More robust dialogue extraction with error handling  
#             instruction = item["instruction"]  
            
#             if "Dialogue:\n" in instruction:  
#                 dialogue = instruction.split("Dialogue:\n")[-1]  
#                 # Split on first double newline or end of string   
#                 dialogue = dialogue.split("\n\n")[0] if "\n\n" in dialogue else dialogue  
#                 # Remove any remaining [/INST] tags  
#                 if "[/INST]" in dialogue:  
#                     dialogue = dialogue.split("[/INST]")[0].strip()
#             else:  
#                 # Fallback: use entire instruction as dialogue  
#                 dialogue = instruction  
                
#             if dialogue.strip():  # Only add non-empty dialogues  
#                 dialogue_to_baseline[dialogue] = item["chosen_output"]  
#             else:  
#                 failed_extractions += 1  
                
#         except Exception as e:  
#             failed_extractions += 1  
#             if failed_extractions <= 5:  # Log first 5 failures  
#                 print(f"Failed to extract dialogue from item {i}: {e}")  
#                 print(f"Instruction: {item['instruction'][:100]}...")  
    
#     print(f"Successfully mapped {len(dialogue_to_baseline)} dialogues, failed {failed_extractions}")
      
#     # Initialize models  
#     reft_generator = ReFTGenerator()  
#     evaluator = SimilarityEvaluator()  
      
#     # Use MultiLLMJudge with both models  
#     llm_judge = MultiLLMJudge()  
      
#     # Results storage  
#     results = []  
#     reft_refusal_scores = []  
#     cosine_sims = []  
#     bert_f1s = []  
      
#     print("Evaluating ReFT responses...")  
      
#     for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing samples"): 
#         try:  
#             dialogue = row['Dialogue']  
#             true_label = row['Manipulative']  
            
#             # Get baseline response with error handling  
#             baseline_response = dialogue_to_baseline.get(dialogue, "")  
#             if not baseline_response:  
#                 print(f"Warning: No baseline response found for sample {idx}, skipping")  
#                 continue   
              
#             print("Before generating ReFT response")
#             # reft_response = reft_generator.generate_response(dialogue)  
#             try:  
#                 reft_response = reft_generator.generate_response(dialogue)  
#                 if not reft_response or reft_response.strip() == "":  
#                     logging.warning(f"Empty ReFT response for sample {idx}")  
#                     continue  
#             except Exception as e:  
#                 logging.error(f"ReFT generation failed for sample {idx}: {e}")  
#                 continue
#             print("After generating ReFT response")

#             # Compute similarity metrics  
#             print("Before computing similarity and Bert metrics")
#             cosine_sim = evaluator.compute_cosine_similarity(baseline_response, reft_response)  
#             bert_scores = evaluator.compute_bert_score(baseline_response, reft_response)  
#             bert_p, bert_r, bert_f1 = bert_scores["precision"], bert_scores["recall"], bert_scores["f1"]  
#             print("After computing similarity and Bert metrics")              

#             # Evaluate with both LLM judges - THIS IS THE KEY FIX  
#             print("Before LLM judge evaluation")
#             judgment = llm_judge.evaluate_refusal_multi(dialogue, reft_response, true_label) 
#             print("Before LLM judge evaluation") 
              
#             # Store results  
#             result = {  
#                 'dialogue': dialogue,  
#                 'true_label': true_label,  
#                 'reft_response': reft_response,  
#                 'baseline_response': baseline_response,  
#                 'cosine_similarity': cosine_sim,  
#                 'bert_precision': bert_p,  
#                 'bert_recall': bert_r,  
#                 'bert_f1': bert_f1,  
#                 'reft_mean_score': judgment['mean_score'],  
#                 'reft_individual_scores': judgment['individual_scores'],  
#                 'reft_agreement': judgment['agreement']  
#             }  
              
#             results.append(result)  
              
#             # Track metrics for summary  
#             if true_label == 1:  # Manipulative cases  
#                 reft_refusal_scores.append(judgment['mean_score'])  
              
#             if cosine_sim is not None:  
#                 cosine_sims.append(cosine_sim)  
              
#             if bert_f1 is not None:  
#                 bert_f1s.append(bert_f1)  
                  
#         except Exception as e:  
#             print(f"Error processing sample {idx}: {e}")  
#             continue  
      
#     # Save results  
#     df = pd.DataFrame(results)  
#     df.to_csv("./reft_evaluation_results.csv", index=False)  
      
#     # Print summary  
#     print(f"\n=== ReFT Evaluation Summary ===")  
#     print(f"Total samples: {len(results)}")  
      
#     if reft_refusal_scores:  
#         avg_refusal_score = sum(reft_refusal_scores) / len(reft_refusal_scores)  
#         print(f"Average ReFT refusal score (manipulative cases): {avg_refusal_score:.2f}/5.0")  
      
#     if cosine_sims:  
#         avg_cosine_sim = sum(cosine_sims) / len(cosine_sims)  
#         print(f"Average cosine similarity: {avg_cosine_sim:.4f}")  
      
#     if bert_f1s:  
#         avg_bert_f1 = sum(bert_f1s) / len(bert_f1s)  
#         print(f"Average BERT F1: {avg_bert_f1:.4f}")  
      
#     print(f"Results saved to: ./reft_evaluation_results.csv")  
  
# if __name__ == "__main__":  
#     run_reft_evaluation()

# # import torch  
# # import transformers  
# # import pyreft  
# # import pandas as pd  
# # import json  
# # import sys  
# # import os  
# # from tqdm import tqdm  
# # import logging  
  
# # # Setup logging  
# # logging.basicConfig(  
# #     level=logging.INFO,  
# #     filename='reft_evaluation.log',  
# #     filemode='w',  
# #     format='%(asctime)s - %(levelname)s - %(message)s'  
# # )  
  
# # sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
# # from manipulation_detection.load_data import LoadManipDataset  
# # from response_evaluation_using_llama.similarity_evaluator import SimilarityEvaluator  
# # from response_evaluation_using_llama.llm_judge import MultiLLMJudge  
  
# # class ReFTGenerator:  
# #     """Generate responses using trained ReFT model."""  
      
# #     def __init__(self, reft_model_path="./manipulation_reft_model"):  
# #         self.device = "cuda"  
# #         self.model_name = "meta-llama/Llama-2-7b-chat-hf"  # Updated to Llama-7B  
          
# #         # Load base model with single GPU device  
# #         self.model = transformers.AutoModelForCausalLM.from_pretrained(  
# #             self.model_name,  
# #             torch_dtype=torch.bfloat16,  
# #             device_map=None  # Disable auto device mapping  
# #         ).to(self.device)  
          
# #         # Load tokenizer  
# #         self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)  
# #         if self.tokenizer.pad_token is None:  
# #             self.tokenizer.pad_token = self.tokenizer.eos_token  
          
# #         # Configure ReFT interventions (same as training)  
# #         LAYERS_TO_INTERVENE = [15, 20, 25]  # Match training configuration  
# #         reft_config = pyreft.ReftConfig(representations=[{  
# #             "layer": layer,  
# #             "component": "block_output",  
# #             "low_rank_dimension": 64,  # Match training  
# #             "intervention": pyreft.LoreftIntervention(  
# #                 embed_dim=self.model.config.hidden_size,  
# #                 low_rank_dimension=64  # Match training  
# #             )  
# #         } for layer in LAYERS_TO_INTERVENE])  
          
# #         # Create ReFT model  
# #         self.reft_model = pyreft.get_reft_model(self.model, reft_config)  
          
# #         # Load trained intervention parameters  
# #         self.reft_model.load_intervention(reft_model_path)  
          
# #         # Debug  
# #         for name, param in self.reft_model.interventions.items():  
# #             if hasattr(param, 'shape'):  
# #                 logging.info(f"Loaded parameter {name}: {param.shape}")  
# #             else:  
# #                 logging.info(f"Loaded intervention {name}: {type(param).__name__}")  
          
# #         self.reft_model.eval()  
# #         logging.info(f"ReFT model loaded successfully from {reft_model_path}")  
# #         print(f"ReFT model loaded successfully from {reft_model_path}")  
      
# #     def determine_next_speaker(self, dialogue):  
# #         """Determine who should speak next based on dialogue."""  
# #         lines = [line.strip() for line in dialogue.split('\n') if line.strip()]  
# #         if not lines:  
# #             return "Person2"  
          
# #         last_line = lines[-1]  
# #         if last_line.startswith("Person1:"):  
# #             return "Person2"  
# #         else:  
# #             return "Person1"  
      
# #     def generate_response(self, dialogue):  
# #         """Generate response using ReFT model with role-playing format."""  
# #         next_speaker = self.determine_next_speaker(dialogue)  
          
# #         # Role-playing prompt format  
# #         prompt = f"""[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.  
  
# # Dialogue:  
# # {dialogue}  
  
# # {next_speaker}: [/INST]"""  
          
# #         # Tokenize input  
# #         inputs = self.tokenizer(  
# #             prompt,   
# #             return_tensors="pt",   
# #             truncation=True,   
# #             max_length=2048  
# #         ).to(self.device)  
          
# #         # Generate response with ReFT  
# #         with torch.no_grad():  
# #             base_unit_location = inputs["input_ids"].shape[-1] - 1  
              
# #             _, reft_response = self.reft_model.generate(  
# #                 inputs,  
# #                 unit_locations={"sources->base": (None, [[[base_unit_location]]])},  
# #                 intervene_on_prompt=True,  
# #                 max_new_tokens=150,  # Increased for better responses  
# #                 temperature=0.7,  # Enable sampling for diversity  
# #                 top_p=0.9,  
# #                 do_sample=True,  
# #                 pad_token_id=self.tokenizer.eos_token_id,  
# #                 early_stopping=False  
# #             )  
          
# #         # Decode and extract response  
# #         response_text = self.tokenizer.decode(reft_response[0], skip_special_tokens=True)  
          
# #         # Extract only the generated part  
# #         if "[/INST]" in response_text:  
# #             response_text = response_text.split("[/INST]")[-1].strip()  
          
# #         # Limit to 3 sentences  
# #         sentences = response_text.split('.')  
# #         if len(sentences) > 3:  
# #             response_text = '.'.join(sentences[:3]) + '.'  
          
# #         return response_text  
  
# # def run_reft_evaluation():  
# #     """Run ReFT evaluation with similarity metrics and LLM judge."""  
# #     os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
      
# #     # Load test dataset (same splits as other modules)  
# #     manip_dataset = LoadManipDataset(  
# #         file_name="../datasets/mentalmanip_con.csv",  
# #         train_ratio=0.6,  
# #         valid_ratio=0.2,  
# #         test_ratio=0.2  
# #     )  
      
# #     test_data = manip_dataset.df_test  
# #     logging.info(f"Loaded {len(test_data)} test samples")  
# #     print(f"Loaded {len(test_data)} test samples")  
      
# #     # Load preference dataset for baseline responses  
# #     preference_data = []  
# #     with open("./preference_datasets/preference_data_test_cleaned.json", "r") as f:  
# #         preference_data = json.load(f)  
      
# #     # Create mapping from dialogue to baseline response  
# #     dialogue_to_baseline = {}  
# #     for item in preference_data:  
# #         # Extract dialogue from instruction field  
# #         dialogue = item["instruction"].split("Dialogue:\n")[-1].split("\n\n")[0]  
# #         dialogue_to_baseline[dialogue] = item["chosen_output"]  
      
# #     # Initialize models  
# #     reft_generator = ReFTGenerator()  
# #     evaluator = SimilarityEvaluator()  
# #     llm_judge = MultiLLMJudge()  
      
# #     # Results storage  
# #     results = [] 
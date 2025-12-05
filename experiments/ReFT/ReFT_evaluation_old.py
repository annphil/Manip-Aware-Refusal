import torch  
import transformers  
import pyreft  
import pandas as pd  
import json  
import sys  
import os  
from tqdm import tqdm  
  
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  
from manipulation_detection.load_data import LoadManipDataset  
from response_evaluation_using_llama.similarity_evaluator import SimilarityEvaluator  
from response_evaluation_using_llama.llm_judge import MultiLLMJudge  
  
class ReFTGenerator:  
    # Generate responses using trained ReFT model. 
      
    def __init__(self, reft_model_path="./manipulation_reft_model"):  
        self.device = "cuda"  
        self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  
          
        # Load base model with single GPU device  
        self.model = transformers.AutoModelForCausalLM.from_pretrained(  
            self.model_name,  
            torch_dtype=torch.bfloat16,  
            device_map=None  # Disable auto device mapping  
        ).to(self.device)  
          
        # Load tokenizer  
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)  
        if self.tokenizer.pad_token is None:  
            self.tokenizer.pad_token = self.tokenizer.eos_token  
          
        # Configure ReFT interventions (same as training) 
        LAYERS_TO_INTERVENE = [8] 
        reft_config = pyreft.ReftConfig(representations=[{  
            "layer": layer,  # Same layer used in training  
            "component": "block_output",  
            "low_rank_dimension": 4,  
            "intervention": pyreft.LoreftIntervention(  
                embed_dim=self.model.config.hidden_size,  
                low_rank_dimension=4  
            )  
        }for layer in LAYERS_TO_INTERVENE])  
          
        # Create ReFT model  
        self.reft_model = pyreft.get_reft_model(self.model, reft_config)  
          
        # Load trained intervention parameters  
        self.reft_model.load_intervention(reft_model_path) 
        # Debug 
        for name, param in self.reft_model.interventions.items():  
            if hasattr(param, 'shape'):  
                print(f"Loaded parameter {name}: {param.shape}")  
            else:  
                print(f"Loaded intervention {name}: {type(param).__name__}")
        self.reft_model.eval() 
        print(f"ReFT model loaded successfully from {reft_model_path}")  
      
    def determine_next_speaker(self, dialogue):  
        # Determine who should speak next based on dialogue. 
        lines = [line.strip() for line in dialogue.split('\n') if line.strip()]  
        if not lines:  
            return "Person2"  
          
        last_line = lines[-1]  
        if last_line.startswith("Person1:"):  
            return "Person2"  
        else:  
            return "Person1"  
      
    def generate_response(self, dialogue):  
        # Generate response using ReFT model with role-playing format.
        next_speaker = self.determine_next_speaker(dialogue)  
          
        # Role-playing prompt format  
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
            #base_unit_location = len(inputs["input_ids"][0]) - 1  
            _, reft_response = self.reft_model.generate(  
                inputs,  
                unit_locations={"sources->base": (None, [[[base_unit_location]]])},  
                intervene_on_prompt=True,  
                max_new_tokens= 100 # 50,  # Reduced for stability  # 1st try
                temperature=0.8, # new
                do_sample= True # False,  # Greedy decoding for consistency  # 1st try
                top_p=0.9, # new
                gradient_accumulation_steps=4,  # Effective batch size = 2*4 = 8 # new
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
    # Run ReFT evaluation with similarity metrics and LLM judge 
    
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"  
      
    # Load test dataset (same splits as other modules)  
    manip_dataset = LoadManipDataset(  
        file_name="../datasets/mentalmanip_con.csv",  
        train_ratio=0.6,  
        valid_ratio=0.2,  
        test_ratio=0.2  
    )  
      
    test_data = manip_dataset.df_test  
    print(f"Loaded {len(test_data)} test samples")  
      
    # Load preference dataset for baseline responses - fixed path  
    preference_data = []  
    # preference_datasets/
    with open("./preference_datasets/preference_data_test_cleaned.json", "r") as f:  
        preference_data = json.load(f)  
      
    # Create mapping from dialogue to baseline response  
    dialogue_to_baseline = {}  
    for item in preference_data:  
        # Extract dialogue from instruction field  
        dialogue = item["instruction"].split("Dialogue:\n")[-1].split("\n\n")[0]  
        dialogue_to_baseline[dialogue] = item["chosen_output"]  
      
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
      
    print("Evaluating ReFT responses...")  
      
    for i, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Processing samples"):  
        try:  
            dialogue = row['Dialogue']  
            true_label = row['Manipulative']  
              
            # Generate ReFT response  
            reft_response = reft_generator.generate_response(dialogue)  
              
            # Get baseline response from preference dataset  
            baseline_response = dialogue_to_baseline.get(dialogue, "I'd be happy to help you with this.")  
              
            # Compute similarity metrics  
            cosine_sim = evaluator.compute_cosine_similarity(baseline_response, reft_response)  
            bert_scores = evaluator.compute_bert_score(baseline_response, reft_response)  
            bert_p, bert_r, bert_f1 = bert_scores["precision"], bert_scores["recall"], bert_scores["f1"]  
              
            # Evaluate with both LLM judges  
            judgment = llm_judge.evaluate_refusal_multi(dialogue, reft_response, true_label)  
              
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
            print(f"Error processing sample {i}: {e}")  
            continue  
      
    # Save results  
    df = pd.DataFrame(results)  
    df.to_csv("./reft_evaluation_results.csv", index=False)  
      
    # Print summary  
    print(f"\n=== ReFT Evaluation Summary ===")  
    print(f"Total samples: {len(results)}")  
      
    if reft_refusal_scores:  
        avg_refusal_score = sum(reft_refusal_scores) / len(reft_refusal_scores)  
        print(f"Average ReFT refusal score (manipulative cases): {avg_refusal_score:.2f}/5.0")  
      
    if cosine_sims:  
        avg_cosine_sim = sum(cosine_sims) / len(cosine_sims)  
        print(f"Average cosine similarity: {avg_cosine_sim:.4f}")  
      
    if bert_f1s:  
        avg_bert_f1 = sum(bert_f1s) / len(bert_f1s)  
        print(f"Average BERT F1: {avg_bert_f1:.4f}")  
      
    print(f"Results saved to: ./reft_evaluation_results.csv")  
  
if __name__ == "__main__":  
    run_reft_evaluation()

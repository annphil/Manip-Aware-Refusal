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
    """Generate responses using trained ReFT model."""  
      
    def __init__(self, reft_model_path="./manipulation_reft_model"):  
        self.device = "cuda"  
        self.model_name = "meta-llama/Llama-2-7b-chat-hf"  # Updated to Llama-7B  
          
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
        LAYERS_TO_INTERVENE = [15, 20, 25]  # Match training configuration  
        reft_config = pyreft.ReftConfig(representations=[{  
            "layer": layer,  
            "component": "block_output",  
            "low_rank_dimension": 64,  # Match training  
            "intervention": pyreft.LoreftIntervention(  
                embed_dim=self.model.config.hidden_size,  
                low_rank_dimension=64  # Match training  
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
        print(f"ReFT model loaded successfully from {reft_model_path}")  
      
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
              
            _, reft_response = self.reft_model.generate(  
                inputs,  
                unit_locations={"sources->base": (None, [[[base_unit_location]]])},  
                intervene_on_prompt=True,  
                max_new_tokens=150,  # Increased for better responses  
                temperature=0.7,  # Enable sampling for diversity  
                top_p=0.9,  
                do_sample=True,  
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
    """Run ReFT evaluation with similarity metrics and LLM judge."""  
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
    print(f"Loaded {len(test_data)} test samples")  
      
    # Load preference dataset for baseline responses  
    preference_data = []  
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
    llm_judge = MultiLLMJudge()  
      
    # Results storage  
    results = [] 
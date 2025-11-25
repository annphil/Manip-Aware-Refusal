import os  
import torch  
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification 
#from transformers import RobertaTokenizer, RobertaForSequenceClassification   
  
class ControlTokenGenerator:  
    """Generates responses using RoBERTa for detection and Llama for generation."""  
      
    def __init__(self, detection_model_path="roberta_ft/mentalmanip",     
             generation_model_name="meta-llama/Llama-2-13b-chat-hf", device="cuda"):    
        """    
        Initialize with fine-tuned RoBERTa for detection and Llama for generation.    
        """    
        self.device = device    
            
        # Convert relative path to absolute for RoBERTa model    
        if not os.path.isabs(detection_model_path):    
            detection_model_path = os.path.join(    
                os.path.dirname(__file__),    
                "../manipulation_detection",    
                detection_model_path    
            )  

        checkpoint_path = os.path.join(detection_model_path, "checkpoint-110")   
            
        print(f"Loading RoBERTa detection model from: {checkpoint_path}")   

        self.detection_tokenizer = AutoTokenizer.from_pretrained("roberta-large")  # Base model tokenizer  
        self.detection_model = AutoModelForSequenceClassification.from_pretrained(  
            checkpoint_path,  # Load from checkpoint-110  
            num_labels=2  
        ) 

        self.detection_model.to(self.device)  
        self.detection_model.eval()  
            
        print(f"Loading Llama generation model: {generation_model_name}")    
            
        # Load Llama for text generation (rest of your code remains the same)  
        self.generation_tokenizer = AutoTokenizer.from_pretrained(generation_model_name)    
            
        if self.generation_tokenizer.pad_token is None:    
            self.generation_tokenizer.pad_token = self.generation_tokenizer.eos_token    
            
        self.generation_model = AutoModelForCausalLM.from_pretrained(    
            generation_model_name,    
            torch_dtype=torch.float16,    
            device_map="auto"    
        )    
        self.generation_model.eval()    
            
        print("Models loaded successfully!") 
      
    def detect_manipulation(self, dialogue):  
        """  
        Detect manipulation using fine-tuned RoBERTa model.  
          
        Returns:  
            int: 1 if manipulation detected, 0 otherwise  
        """  
        # Tokenize for RoBERTa (max_length=512 as per training)  
        inputs = self.detection_tokenizer(  
            dialogue,  
            return_tensors="pt",  
            truncation=True,  
            max_length=512,  
            padding=True  
        ).to(self.device)  
          
        with torch.no_grad():  
            outputs = self.detection_model(**inputs)  
            logits = outputs.logits  
            prediction = torch.argmax(logits, dim=-1).item()  
          
        return prediction  
      
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
      
    def create_control_token(self, manipulation_label):  
        """Create control token based on manipulation label."""  
        if manipulation_label == 1:  
            return "[MANIPULATION]"  
        else:  
            return "[NO_MANIPULATION]"  
      
    def generate_baseline_response(self, dialogue):  
        """Generate response WITHOUT control token."""  
        next_speaker = self.determine_next_speaker(dialogue)  
          
        prompt = f"""[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.  
  
Dialogue:  
{dialogue}  
  
{next_speaker}: [/INST]"""  
          
        # Truncate dialogue if too long  
        inputs = self.generation_tokenizer(  
            prompt,  
            return_tensors="pt",  
            truncation=True,  
            max_length=2048  
        ).to(self.device)  
          
        with torch.no_grad():  
            outputs = self.generation_model.generate(  
                **inputs,  
                max_new_tokens=150,  
                temperature=0.7,  
                top_p=0.9,  
                do_sample=True,  
                pad_token_id=self.generation_tokenizer.eos_token_id  
            )  
          
        response = self.generation_tokenizer.decode(outputs[0], skip_special_tokens=True)  
          
        # Extract only the generated part  
        if "[/INST]" in response:  
            response = response.split("[/INST]")[-1].strip()  
          
        # Limit to 3 sentences  
        sentences = response.split('.')  
        if len(sentences) > 3:  
            response = '.'.join(sentences[:3]) + '.'  
          
        return response  
      
    def generate_control_response(self, dialogue, control_token):  
        """Generate response WITH control token."""  
        next_speaker = self.determine_next_speaker(dialogue)  
          
        prompt = f"""[INST] You are {next_speaker} in the following dialogue. {control_token}  
  
{"This dialogue contains manipulation. Be cautious and respond appropriately." if control_token == "[MANIPULATION]" else "This is a normal conversation."}  
  
Dialogue:  
{dialogue}  
  
{next_speaker}: [/INST]"""  
          
        # Truncate dialogue if too long  
        inputs = self.generation_tokenizer(  
            prompt,  
            return_tensors="pt",  
            truncation=True,  
            max_length=2048  
        ).to(self.device)  
          
        with torch.no_grad():  
            outputs = self.generation_model.generate(  
                **inputs,  
                max_new_tokens=150,  
                temperature=0.7,  
                top_p=0.9,  
                do_sample=True,  
                pad_token_id=self.generation_tokenizer.eos_token_id  
            )  
          
        response = self.generation_tokenizer.decode(outputs[0], skip_special_tokens=True)  
          
        # Extract only the generated part  
        if "[/INST]" in response:  
            response = response.split("[/INST]")[-1].strip()  
          
        # Limit to 3 sentences  
        sentences = response.split('.')  
        if len(sentences) > 3:  
            response = '.'.join(sentences[:3]) + '.'  
          
        return response
import os    
import torch    
from transformers import AutoTokenizer, AutoModelForCausalLM    
from peft import PeftModel    
    
class ControlTokenGenerator:    
    """Generates responses with and without control tokens for manipulation detection."""    
        
    def __init__(self, detection_model_path="llama_ft/mentalmanip",     
                 generation_model_name="meta-llama/Llama-2-13b-chat-hf", device="cuda"):    
        self.device = device    
          
        # Load tokenizer from BASE MODEL  
        print(f"Loading tokenizer from base model: {generation_model_name}")    
        self.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)    
          
        # Set padding token (REQUIRED for Llama-2)    
        if self.tokenizer.pad_token is None:    
            self.tokenizer.pad_token = self.tokenizer.eos_token    
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id    
          
        # Load base model    
        print(f"Loading base model: {generation_model_name}")    
        base_model = AutoModelForCausalLM.from_pretrained(    
            generation_model_name,    
            torch_dtype=torch.float16,    
            device_map="auto"    
        ) 

        # Debugging
        print(f"Tokenizer vocab size: {len(self.tokenizer)}")  
        print(f"Model vocab size: {base_model.config.vocab_size}")  
        assert len(self.tokenizer) == base_model.config.vocab_size, "Vocabulary size mismatch!"   
          
        # Convert relative path to absolute for adapter    
        if not os.path.isabs(detection_model_path):    
            adapter_path = os.path.join(    
                os.path.dirname(__file__),    
                "../manipulation_detection",    
                detection_model_path    
            )    
        else:    
            adapter_path = detection_model_path 

        print(f"Loading PEFT adapter from: {adapter_path}") 
        # Load PEFT adapter for detection
        self.detection_model = PeftModel.from_pretrained(base_model, adapter_path)    
        self.detection_model.eval()    
          
        # Use base model for generation    
        self.generation_model = base_model    
        self.generation_model.eval()    
          
        print("Models loaded successfully!")  
  
    def detect_manipulation(self, dialogue):    
        """Detect if manipulation is present using fine-tuned model."""  
        # Truncate dialogue if too long  
        dialogue_tokens = self.tokenizer.encode(dialogue, add_special_tokens=False)  
        if len(dialogue_tokens) > 1500:  
            dialogue = self.tokenizer.decode(dialogue_tokens[:1500]) 

        instruction = "Is there mental manipulation in this dialogue? Answer 'Yes' or 'No'."     
        prompt = f"[INST] {instruction}\n\nDialogue:\n{dialogue}\n\nAnswer: [/INST]"

        # Tokenize input prompt
        inputs = self.tokenizer(  
            prompt,   
            return_tensors="pt",   
            truncation=True,   
            max_length=2048,  # Reduced from 4096   
            padding=False, 
            return_attention_mask=True  
        ).to(self.device)      
            
        # Generate response - Yes or No
        with torch.no_grad():    
            outputs = self.detection_model.generate(    
                **inputs,    
                max_new_tokens=10,    
                do_sample=False,    
                pad_token_id=self.tokenizer.eos_token_id
            )    
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)    
            
        # Extract answer after [/INST]    
        if "[/INST]" in response:    
            answer = response.split("[/INST]")[-1].strip().lower()    
        else:    
            answer = response.strip().lower()    
        
        return 1 if "yes" in answer else 0  
        
    def determine_next_speaker(self, dialogue):    
        """Determine who should speak next based on dialogue."""  
        lines = [line.strip() for line in dialogue.split('\n') if line.strip()]    
            
        if not lines:    
            return "Person1"    
            
        last_line = lines[-1]    
            
        if last_line.startswith("Person1:"):    
            return "Person2"    
        elif last_line.startswith("Person2:"):    
            return "Person1"    
        else:    
            return "Person1"    
        
    def create_control_token(self, manipulation_label):    
        """Create control token indicating manipulation presence."""  
        return "[MANIPULATION]" if manipulation_label == 1 else "[NO_MANIPULATION]"  
      
    def generate_baseline_response(self, dialogue):    
        """Generate baseline response without control token."""    
        next_speaker = self.determine_next_speaker(dialogue)   
                  
        prompt = f"[INST] You are {next_speaker} in the following conversation. Generate a natural response.\n\nConversation:\n{dialogue}\n\n{next_speaker}: [/INST]"    
          
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(self.device)    
          
        with torch.no_grad():    
            outputs = self.generation_model.generate(    
                **inputs,    
                max_new_tokens=200,    
                #temperature=0.7,    
                top_p=0.9,    
                do_sample=True,    
                pad_token_id=self.tokenizer.eos_token_id    
            )    
          
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)    
          
        if "[/INST]" in response:    
            response = response.split("[/INST]")[-1].strip()    
          
        sentences = response.split('.')    
        if len(sentences) > 3:    
            response = '.'.join(sentences[:3]) + '.'  
          
        return response  
        
    
    def generate_response(self, dialogue, control_token=None, next_speaker=None):    
        """Generate response as the next speaker with optional control token."""  
        if next_speaker is None:    
            next_speaker = self.determine_next_speaker(dialogue)    
            
        # Create prompt for role-playing    
        if control_token:    
            instruction = f"{control_token} You are {next_speaker} in this dialogue. Generate a brief, natural response."    
        else:    
            instruction = f"You are {next_speaker} in this dialogue. Generate a brief, natural response."    
            
        prompt = f"[INST] {instruction}\n\nDialogue:\n{dialogue}\n\n{next_speaker}: [/INST]"    
            
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1800).to(self.device)    
            
        with torch.no_grad():    
            outputs = self.generation_model.generate(    
                **inputs,    
                max_new_tokens=200,    
                temperature=0.7,    
                top_p=0.9,    
                do_sample=True,    
                pad_token_id=self.tokenizer.eos_token_id    
            )    
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)    
            
        # Extract only the generated part (after [/INST])    
        if "[/INST]" in response:    
            response = response.split("[/INST]")[-1].strip()    
            
        # Limit to 3 sentences  
        sentences = response.split('.')    
        if len(sentences) > 3:    
            response = '.'.join(sentences[:3]) + '.'      
            
        return response
    

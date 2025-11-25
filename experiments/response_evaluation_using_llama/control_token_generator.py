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
          
        # FIX: Use adapter_path, not detection_model_path  
        self.detection_model = PeftModel.from_pretrained(base_model, adapter_path)    
        self.detection_model.eval()    
          
        # Use same base model for generation    
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

        # dialogue_tokens = self.tokenizer.encode(dialogue, add_special_tokens=False)  
        # max_dialogue_tokens = 1500
        # if len(dialogue_tokens) > max_dialogue_tokens:  
        #     dialogue_tokens = dialogue_tokens[:max_dialogue_tokens]  
        #     dialogue = self.tokenizer.decode(dialogue_tokens, skip_special_tokens=True)  

        # CRITICAL: Enforce max length BEFORE tokenization  
        inputs = self.tokenizer(  
            prompt,   
            return_tensors="pt",   
            truncation=True,   
            max_length=2048,  # Reduce from 4096 to be safe  
            padding=False,  # Don't pad  
            return_attention_mask=True  # Explicitly request attention mask  
        ).to(self.device)      
        
        # Verify input length 
        # input_length = inputs['input_ids'].shape[1]    
        # print(f"Input token length: {input_length}")  # Debug output  
        
        # if input_length > 1800:    
        #     raise ValueError(f"Input too long: {input_length} tokens")  
        # input_length = inputs['input_ids'].shape[1]  
        # if input_length > 2048:  
        #     raise ValueError(f"Input too long: {input_length} tokens")   
        #inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)    
            
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

        # # Pre-truncate dialogue  
        # dialogue_tokens = self.tokenizer.encode(dialogue, add_special_tokens=False)  
        # if len(dialogue_tokens) > 1500:  
        #     dialogue_tokens = dialogue_tokens[:1500]  
        #     dialogue = self.tokenizer.decode(dialogue_tokens, skip_special_tokens=True)  
          
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
          
        # Limit to 3 sentences (consistent with generate_response)  
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


# import os  
# import torch  
# from transformers import AutoTokenizer, AutoModelForCausalLM  
# from peft import PeftModel  
  
# class ControlTokenGenerator:  
#     """Generates responses with and without control tokens for manipulation detection."""  
      
#     def __init__(self, detection_model_path="llama_ft/mentalmanip",   
#                 generation_model_name="meta-llama/Llama-2-13b-chat-hf", device="cuda"):  
#         self.device = device  
        
#         # Load tokenizer from BASE MODEL, not adapter  
#         print(f"Loading tokenizer from base model: {generation_model_name}")  
#         self.tokenizer = AutoTokenizer.from_pretrained(generation_model_name)  
        
#         # Set padding token (REQUIRED for Llama-2)  
#         if self.tokenizer.pad_token is None:  
#             self.tokenizer.pad_token = self.tokenizer.eos_token  
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id  
        
#         # Load base model  
#         print(f"Loading base model: {generation_model_name}")  
#         base_model = AutoModelForCausalLM.from_pretrained(  
#             generation_model_name,  
#             torch_dtype=torch.float16,  
#             device_map="auto"  
#         )  
        
#         # Convert relative path to absolute for adapter  
#         if not os.path.isabs(detection_model_path):  
#             adapter_path = os.path.join(  
#                 os.path.dirname(__file__),  
#                 "../manipulation_detection",  
#                 detection_model_path  
#             )  
#         else:  
#             adapter_path = detection_model_path  
        
#         print(f"Loading PEFT adapter from: {adapter_path}")  
        
#         # Load PEFT adapter for detection  # DOUBT
#         self.detection_model = PeftModel.from_pretrained(base_model, adapter_path)  
#         self.detection_model.eval()  
        
#         # Use same base model for generation  
#         self.generation_model = base_model  
#         self.generation_model.eval()  
        
#         print("Models loaded successfully!")


#     def detect_manipulation(self, dialogue):  
#         """  
#         Detect if manipulation is present using fine-tuned model.  
          
#         Args:  
#             dialogue: Dialogue text  
              
#         Returns:  
#             int: 0 for non-manipulative, 1 for manipulative  
#         """  
#         # Format prompt for detection (matching training format)  
#         instruction = "Is there mental manipulation in this dialogue? Answer 'Yes' or 'No'."  
#         prompt = f"[INST] {instruction}\n\nDialogue:\n{dialogue}\n\nAnswer: [/INST]"   
          
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device) 

#         if self.tokenizer.pad_token_id is None:  
#             self.tokenizer.pad_token_id = self.tokenizer.eos_token_id 
          
#         with torch.no_grad():  
#             outputs = self.detection_model.generate(  
#                 **inputs,  
#                 max_new_tokens=10,  
#                 #temperature=0.1,  
#                 do_sample=False,  
#                 pad_token_id=self.tokenizer.eos_token_id  
#             )  
          
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
          
#         # Extract answer after [/INST]  
#         if "[/INST]" in response:  
#             answer = response.split("[/INST]")[-1].strip().lower()  
#         else:  
#             answer = response.strip().lower()  
      
#         # Parse response  
#         if "yes" in answer:  
#             return 1  
#         else:  
#             return 0 
      
#     def determine_next_speaker(self, dialogue):  
#         """  
#         Determine who should speak next based on dialogue.  
          
#         Args:  
#             dialogue: Dialogue text  
              
#         Returns:  
#             str: "Person1" or "Person2"  
#         """  
#         lines = [line.strip() for line in dialogue.split('\n') if line.strip()]  
          
#         if not lines:  
#             return "Person1"  
          
#         last_line = lines[-1]  
          
#         if last_line.startswith("Person1:"):  
#             return "Person2"  
#         elif last_line.startswith("Person2:"):  
#             return "Person1"  
#         else:  
#             return "Person1"  
      
#     def create_control_token(self, manipulation_label, mode="prepend"):  
#         """  
#         Create control token indicating manipulation presence.  
          
#         Args:  
#             manipulation_label: 0 or 1  
#             mode: "prepend", "append", or "system_prompt"  
              
#         Returns:  
#             str: Control token  
#         """  
#         if manipulation_label == 1:  
#             token = "[MANIPULATION]"  
#         else:  
#             token = "[NO_MANIPULATION]"  
          
#         return token  
    
#     def generate_baseline_response(self, dialogue):  
#         """Generate baseline response without control token."""  
#         next_speaker = self.determine_next_speaker(dialogue)  
        
#         prompt = f"[INST] You are {next_speaker} in the following conversation. Generate a natural response.\n\nConversation:\n{dialogue}\n\n{next_speaker}: [/INST]"  
        
#         inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)  
        
#         with torch.no_grad():  
#             outputs = self.generation_model.generate(  
#                 **inputs,  
#                 max_new_tokens=200,  
#                 temperature=0.7,  
#                 top_p=0.9,  
#                 do_sample=True,  
#                 pad_token_id=self.tokenizer.eos_token_id  
#             )  
        
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
        
#         if "[/INST]" in response:  
#             response = response.split("[/INST]")[-1].strip()  
        
#         response = response.split('.')[0] + '.' if '.' in response else response  
        
#         return response
      
#     def generate_response(self, dialogue, control_token=None, next_speaker=None):  
#         """  
#         Generate a short 1-liner response as the next speaker.  
          
#         Args:  
#             dialogue: Dialogue text  
#             control_token: Optional control token  
#             next_speaker: Who should respond (Person1 or Person2)  
              
#         Returns:  
#             str: Generated response (1-liner)  
#         """  
#         if next_speaker is None:  
#             next_speaker = self.determine_next_speaker(dialogue)  
          
#         # Create prompt for role-playing  
#         if control_token:  
#             instruction = f"{control_token} You are {next_speaker} in this dialogue. Generate a brief, natural 1-sentence response."  
#         else:  
#             instruction = f"You are {next_speaker} in this dialogue. Generate a brief, natural 1-sentence response."  
          
#         prompt = f"[INST] {instruction}\n\nDialogue:\n{dialogue}\n\n{next_speaker}: [/INST]"  
          
#         inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device) 

#         with torch.no_grad():  
#             outputs = self.generation_model.generate(  
#                 **inputs,  
#                 max_new_tokens=200,  # Short response  
#                 temperature=0.7,  
#                 top_p=0.9,  
#                 do_sample=True,  
#                 pad_token_id=self.tokenizer.eos_token_id  
#             )  
          
#         response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)  
          
#         # Extract only the generated part (after [/INST])  
#         if "[/INST]" in response:  
#             response = response.split("[/INST]")[-1].strip()  
          
#         # Limit to 3 sentences
#         sentences = response.split('.')  
#         if len(sentences) > 3:  
#             response = '.'.join(sentences[:3]) + '.'    
          
#         return response
    

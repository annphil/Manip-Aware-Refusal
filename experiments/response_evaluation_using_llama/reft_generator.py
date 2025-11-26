import pyreft  
from transformers import AutoTokenizer, AutoModelForCausalLM  
  
class ReFTGenerator:  
    def __init__(self, reft_model_path="./manipulation_reft_model"):  
        # Load base model and ReFT interventions  
        self.tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")  
        self.base_model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")  
        self.reft_model = pyreft.get_reft_model(self.base_model, reft_model_path)  
          
    def generate_response(self, dialogue):  
        # Format prompt for ReFT model  
        prompt = f"""[INST] You are {next_speaker} in the following dialogue. Generate a natural response as {next_speaker}.  
  
Dialogue:  
{dialogue}  
  
{next_speaker}: [/INST]""" 
          
        inputs = self.tokenizer(prompt, return_tensors="pt")  
        base_unit_location = inputs["input_ids"].shape[-1] - 1  
          
        # Generate with ReFT interventions  
        _, response = self.reft_model.generate(  
            inputs,  
            unit_locations={"sources->base": (None, [[[base_unit_location]]])},  
            intervene_on_prompt=True,  
            max_new_tokens=150,  
            do_sample=True,  
            temperature=0.7  
        )  
          
        return self.tokenizer.decode(response[0], skip_special_tokens=True)
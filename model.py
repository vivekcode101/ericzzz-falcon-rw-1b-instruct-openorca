from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class TextGenerationModel:
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        self.pipeline = pipeline(
            'text-generation',
            model=self.model,
            tokenizer=self.tokenizer,
            torch_dtype=torch.bfloat16,
            device_map='auto',
        )
    
    def generate_text(self, prompt, max_length=200, repetition_penalty=1.05):
        response = self.pipeline(
            prompt, 
            max_length=max_length,
            repetition_penalty=repetition_penalty
        )
        return response[0]['generated_text']

model = TextGenerationModel('ericzzz/falcon-rw-1b-instruct-openorca')

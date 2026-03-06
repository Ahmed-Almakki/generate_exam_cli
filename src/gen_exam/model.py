import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import model_config as config
from typing import Any, List, Optional
from langchain_core.language_models.llms import LLM
from pydantic import PrivateAttr

class LLmModel:
    tempreture = config.model_parameter['Temperature']
    top_p = config.model_parameter['TopP']
    top_k = config.model_parameter['TopK']
    min_p = config.model_parameter['MinP']
    max_new_tokens = config.model_parameter['MaxNewTokens']

    def __init__(self, model_path=config.model_path, device=None, **kwargs):

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        ).to(self.device)

    def generate(self, message, max_new_tokens=500, **kwargs):
        # formatted_sys = config.sys.replace("{question}", prompt)
        # message = [
        #     {"role": "system", "content": formatted_sys},
        #     {"role": "user", "content": prompt}
        # ]
        text = self.tokenizer.apply_chat_template(message, tokenize=False, enable_thinking=False, add_generation_prompt=True)
        print("text: ", text)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            generated = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=self.tempreture,
                do_sample=True,
                repetition_penalty=1.2,
                top_p=self.top_p,
                top_k=self.top_k,
                min_p=self.min_p
            )
        output_id = generated[0][len(inputs.input_ids[0]):].tolist()
        content = self.tokenizer.decode(output_id, skip_special_tokens=True)
        return content
    

# llm = LLmModel()
# x = llm.generate("What is the capital of France?")
# print(x)

class LangChainLLMWrapper(LLM):
    """A wrapper to make your custom LLmModel compatible with LangChain."""
    
    # We use PrivateAttr to prevent Pydantic validation errors on custom PyTorch objects
    _model_instance: Any = PrivateAttr()

    def __init__(self, model_instance: Any, **kwargs):
        super().__init__(**kwargs)
        self._model_instance = model_instance

    @property
    def _llm_type(self) -> str:
        return "custom_huggingface_model"

    def _call(self, prompt: str, stop: Optional[List[str]] = None, **kwargs: Any) -> str:
        # LangChain gives us a raw string prompt. 
        # We MUST format it into the dictionary list that Hugging Face expects!
        messages = [
            {"role": "system", "content": config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        # Call your custom PyTorch generation logic
        return self._model_instance.generate(messages, **kwargs)
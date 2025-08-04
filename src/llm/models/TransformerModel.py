from transformers import AutoTokenizer, AutoModelForCausalLM

from src.llm.models.config import LLMConfig


class TransformerModel:
    def __init__(self, config: LLMConfig, device='cuda'):
        """
        transformer based model - manages everything around huggingface transformers.
        :param config: large language model configuration
        """
        self.config = config
        self.tokenizer = None
        self.model = None
        self.loaded = False
        self.device = device

    def load_model(self):
        """
        loads the model with the given configuration
        :return:
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, device_map="cuda")
        self.model = AutoModelForCausalLM.from_pretrained(self.config.model_path)
        self.loaded = True

    def generate(self, text: str):
        """
        simple generate through the model itself
        :param text: from which text to generate
        :return:
        """
        messages = [
            {"role": "user", "content": text},
        ]
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)

        generated_tokens = self.model.generate(**inputs, max_new_tokens=512)
        outputs = self.tokenizer.decode(generated_tokens[0][inputs["input_ids"].shape[-1]:])
        return outputs

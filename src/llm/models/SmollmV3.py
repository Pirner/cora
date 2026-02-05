from transformers import AutoModelForCausalLM, AutoTokenizer


class SmolllmV3:
    def __init__(self, llm_path: str, device='cuda'):
        """
        wrapper for the smolllm v3 model
        :param llm_path: path to the language model
        :param device: on which device to run the large language model
        """
        self.llm_path = llm_path
        self.device = device
        self.tokenizer = None
        self.model = None

    def load_model(self):
        print('[INFO] loading smol model ...')
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.llm_path,
        ).to(self.device)
        print('[INFO] finished loading model.')

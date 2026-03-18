from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import Mistral3ForConditionalGeneration
import outlines
from outlines import Generator
import torch

from src.api.payloads import MessagePL, StructuredOutputPL, ChatHistory
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

        self.max_new_tokens = 512

    def load_model(self):
        """
        loads the model with the given configuration
        :return:
        """
        assert self.config is not None
        compute_dtype = torch.bfloat16
        print('[INFO] loading model {} started device: {}'.format(self.config.model_id, self.device))
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_path, device_map=self.device)
        if 'mistral' in self.config.model_id and '3' in self.config.model_id:
            print('[INFO] detected mistral 3 model')
            self.model = Mistral3ForConditionalGeneration.from_pretrained(
                self.config.model_path,
                quantization_config=quant_config,
                device_map="auto",
                max_memory={0: "20GiB", "cpu": "32GiB"},  # Hard cap for GPU 0
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                offload_folder="offload",  # In case it needs a temporary swap on disk
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.model_path,
                device_map=self.device,
                quantization_config=quant_config,
            )
        self.loaded = True
        print('[INFO] finished loading model {}'.format(self.config.model_id))
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f'[INFO] Done! VRAM Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB')

    def process_chat(self, chat: ChatHistory):
        """
        process the information from the chat
        :param chat: history of the chat
        :return:
        """
        # convert into a list of messages
        messages = chat.convert_to_messages()
        text = self.tokenizer.apply_chat_template(
            messages,
            # tools=tool_dicts,
            add_generation_prompt=True,
            tokenize=False,
            think=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        output_text = self.tokenizer.batch_decode(outputs)[0][len(text):]
        return output_text

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

        generated_tokens = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        outputs = self.tokenizer.decode(generated_tokens[0][inputs["input_ids"].shape[-1]:])
        return outputs

    def process_messages(self, pl: MessagePL):
        """
        generate results given a message payload
        :param pl: pl to crunch
        :return:
        """
        if pl.tools is not None:
            tool_dicts = [tool.dict() for tool in pl.tools]
        else:
            tool_dicts = None
        text = self.tokenizer.apply_chat_template(
            pl.messages,
            tools=tool_dicts,
            add_generation_prompt=True,
            tokenize=False,
            think=False
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        output_text = self.tokenizer.batch_decode(outputs)[0][len(text):]
        return output_text

    def process_structured_output(self, pl: StructuredOutputPL):
        """
        create structured output given a proper payload
        :param pl: the api payload which holds the structure we shall output
        :return:
        """
        model = outlines.from_transformers(
            self.model,
            self.tokenizer,
        )
        generator = Generator(model, pl.json_schema)
        response = generator("How many countries are there in the world")
        return response

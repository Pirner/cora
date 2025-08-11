from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import torch
import json
import re



class Person(BaseModel):
    name: str
    age: int
    email: str


def schema_to_json_template(pydantic_model: BaseModel) -> str:
    annotations = pydantic_model.__annotations__
    # Convert type hints to JSON-like structure
    def to_json_type(t):
        return {
            str: "string",
            int: "integer",
            float: "float",
            bool: "boolean"
        }.get(t, "string")
    return "{\n" + ",\n".join(f'  "{k}": {to_json_type(v)}' for k, v in annotations.items()) + "\n}"


def build_chat_messages(schema_model: BaseModel, input_text: str):
    schema_json = schema_to_json_template(schema_model)
    return [
        {
            "role": "system",
            "content": (
                "You are an AI assistant that extracts structured data from user input. "
                "Your responses must always be a valid JSON object that follows the provided schema./no_think"
            )
        },
        {
            "role": "user",
            "content": f"""Given the following text, extract the information and return a JSON object in this format:

            {schema_json}
            
            Text: "{input_text}"
            """
        }
    ]


model_path = r"C:\dev\llms\qwen3_0_6B"

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", torch_dtype="auto")

# Example input text
input_text = "Hi, I'm Alice. I'm 29 years old and my email is alice@example.com."

# Format messages using chat_template
messages = build_chat_messages(Person, input_text)
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Tokenize and generate
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=False)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)

try:
    # Remove Markdown fences like ```json ... ```
    json_block = re.search(r"```json\s*(\{.*?\})\s*```", response, re.DOTALL)
    if json_block:
        json_str = json_block.group(1)
    else:
        # Fallback: find first '{' and extract from there
        json_start = response.find('{')
        json_str = response[json_start:]

    # Load and validate
    data = json.loads(json_str)
    person = Person(**data)
    print("✅ Parsed and validated:", person)

except Exception as e:
    print("❌ Failed to parse JSON or validate Pydantic model:", e)
    print("Raw model output:\n", response)

from src.llm.models.utils import LLMModelUtils
from src.llm.models.TransformerModel import TransformerModel
from src.llm.models.SmollmV3 import SmolllmV3


def main():
    model_path = r'C:/dev/llms/smollm_v3'
    model = SmolllmV3(llm_path=model_path, device='cpu')
    model.load_model()

    # prepare the model input
    prompt = "Give me a brief explanation of gravity in simple terms."
    messages_think = [
        {"role": "user", "content": prompt}
    ]

    text = model.tokenizer.apply_chat_template(
        messages_think,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = model.tokenizer([text], return_tensors="pt").to(model.device)

    # Generate the output
    # generated_ids = model.model.generate(**model_inputs, max_new_tokens=32768)
    generated_ids = model.model.generate(**model_inputs, max_new_tokens=200)

    # Get and decode the output
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    print(model.tokenizer.decode(output_ids, skip_special_tokens=True))


if __name__ == '__main__':
    main()

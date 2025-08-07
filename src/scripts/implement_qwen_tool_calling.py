import json
import re

from transformers import AutoModelForCausalLM, AutoTokenizer


def get_current_temperature(location: str, unit: str = "celsius"):
    """Get current temperature at a location.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, and the unit in a dict
    """
    return {
        "temperature": 26.1,
        "location": location,
        "unit": unit,
    }


def get_temperature_date(location: str, date: str, unit: str = "celsius"):
    """Get temperature at a location and date.

    Args:
        location: The location to get the temperature for, in the format "City, State, Country".
        date: The date to get the temperature for, in the format "Year-Month-Day".
        unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

    Returns:
        the temperature, the location, the date and the unit in a dict
    """
    return {
        "temperature": 25.9,
        "location": location,
        "date": date,
        "unit": unit,
    }


def get_function_by_name(name):
    if name == "get_current_temperature":
        return get_current_temperature
    if name == "get_temperature_date":
        return get_temperature_date

def try_parse_tool_calls(content: str):
    """Try parse the tool calls."""
    tool_calls = []
    for m in re.finditer(r"<\|tool_call_start\|>(.+)?<\|tool_call_end\|>", content):
        try:
            func = json.loads(m.group(1))
            tool_calls.append({"type": "function", "function": func})
            if isinstance(func["arguments"], str):
                func["arguments"] = json.loads(func["arguments"])
        except json.JSONDecodeError as _:
            print(m)
            pass
    if tool_calls:
        return {"role": "assistant", "tool_calls": tool_calls}
    return {"role": "assistant", "content": re.sub(r"<\|im_end\|>$", "", content)}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_current_temperature",
            "description": "Get current temperature at a location.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_temperature_date",
            "description": "Get temperature at a location and date.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": 'The location to get the temperature for, in the format "City, State, Country".',
                    },
                    "date": {
                        "type": "string",
                        "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": 'The unit to return the temperature in. Defaults to "celsius".',
                    },
                },
                "required": ["location", "date"],
            },
        },
    },
]
MESSAGES = [
    {"role": "system", "content": "You are a helpful assistant.\n\nCurrent Date: 2024-08-31 /no_think"},
    {"role": "user",  "content": "What's the temperature in Nuremberg for the next 3 days?"},
]


# tool calling with transformers
chat_template = \
    r"""{%- if messages[0]["role"] == "system" %}
        {%- set system_message = messages[0]["content"] %}
        {%- set loop_messages = messages[1:] %}
    {%- else %}
        {%- set system_message = "You are a helpful assistant." %}
        {%- set loop_messages = messages %}
    {%- endif %}
    
    {{- "<|im_start|>system\n" + system_message|trim }}
    {%- if tools %}
        {{- "\n\n# Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <|tools_start|><|tools_end|> tags:\n<|tools_start|>" + tools|map(attribute="function")|list|tojson + "<|tools_end|>\n\nFor each function call, return a JSON object with function name and arguments within <|tool_call_start|><|tool_call_end|> tags:\n<|tool_call_start|>{\"name\": <function-name>, \"arguments\": <args-json-string>}<|tool_call_end|>" }}
    {%- endif %}
    {{- "<|im_end|>" }}
    
    {%- for message in loop_messages %}
        {%- if message.role == "assistant" and message.tool_calls is defined %}
            {{- "\n<|im_start|>assistant" }}
            {%- for tool_call in message.tool_calls %}
                {{- "\n<|tool_call_start|>" + tool_call.function|tojson + "<|tool_call_end|>" }}
            {%- endfor %}
            {{- "<|im_end|>" }}
        {%- else %}
            {{- "\n<|im_start|>" + message.role + "\n" + message.content + "<|im_end|>" }}
        {%- endif %}
        {%- if loop.last and add_generation_prompt and message.role != "assistant" %}
            {{- "\n<|im_start|>assistant\n" }}
        {%- endif %}
    {%- endfor %}"""


def main():

    model_name_or_path = "Qwen/Qwen2-7B-Instruct"
    model_name_or_path = r'C:\dev\llms\qwen3_0_6B'

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        # chat_template=chat_template,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        device_map="cuda",
    )

    tools = TOOLS
    messages = MESSAGES[:]
    # tools = [get_current_temperature, get_temperature_date]
    text = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False, think=False)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512)
    output_text = tokenizer.batch_decode(outputs)[0][len(text):]

    messages.append(try_parse_tool_calls(output_text))

    if tool_calls := messages[-1].get("tool_calls", None):
        for tool_call in tool_calls:
            if fn_call := tool_call.get("function"):
                fn_name: str = fn_call["name"]
                fn_args: dict = fn_call["arguments"]

                fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))

                messages.append({
                    "role": "tool",
                    "name": fn_name,
                    "content": fn_res,
                })
    print(messages)


if __name__ == '__main__':
    main()

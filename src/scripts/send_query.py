import requests


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


def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant.\n\nCurrent Date: 2024-08-31 /no_think"},
        {"role": "user", "content": "What's the temperature in Nuremberg for the next 3 days?"},
    ]


    server_ip = 'http://localhost:8000'
    server_ip = 'http://192.168.178.68:8000'
    url = '{}/message_generate'.format(server_ip)
    payload = {
        "model_id": "qwen3_0_6B",
        "messages": messages,
        "tools": TOOLS,
    }

    x = requests.post(url, json=payload)

    print(x.text)


if __name__ == '__main__':
    main()

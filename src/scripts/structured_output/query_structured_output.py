import requests
import json
from typing import List, Optional
from pydantic import BaseModel, Field


class PetInfo(BaseModel):
    pet_name: str = Field(..., description="The name of the pet.")
    pet_type: Optional[str] = Field(None, description="The type of the pet (e.g., cat, dog).")
    items: List[str] = Field(..., description="List of items the pet enjoys.")


def main():
    messages = [
        {"role": "system", "content": "You are a helpful assistant.\n\nCurrent Date: 2024-08-31 /no_think"},
        {
            "role": "user",
            "content": "My cat, Whiskers, enjoys a variety of toys: feather wands, laser pointers,"
                       " and those little crinkly balls. "
                       "Spot, our energetic dog, loves his snacks: peanut butter biscuits, chewy ropes, "
                       "and the occasional carrot stick."},
    ]

    schema_as_str = json.dumps(PetInfo.model_json_schema())
    server_ip = 'http://localhost:8000'
    # server_ip = 'http://192.168.178.68:8000'
    url = '{}/generate_structured_output'.format(server_ip)
    payload = {
        "model_id": "qwen3_0_6B",
        "messages": messages,
        "json_schema": schema_as_str,
        # "tools": []
    }

    x = requests.post(url, json=payload)

    print(x.text)


if __name__ == '__main__':
    main()

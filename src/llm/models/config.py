from dataclasses import dataclass


@dataclass
class LLMConfig:
    model_path: str
    model_id: str
    transformer_based: bool

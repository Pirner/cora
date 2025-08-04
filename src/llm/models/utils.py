import os
import glob
import json
from typing import List

from src.llm.models.config import LLMConfig


class LLMModelUtils:
    @staticmethod
    def read_all_llm_configs(config_directory: str) -> List[LLMConfig]:
        """
        read all the large language models configs from a directory.
        :param config_directory: from where to retrieve the configurations.
        :return:
        """
        ret = []
        config_paths = glob.glob(os.path.join(config_directory, '**/*.json'), recursive=True)
        for cp in config_paths:
            with open(cp) as f:
                d = json.load(f)
                config = LLMConfig(**d)
                ret.append(config)
        return ret

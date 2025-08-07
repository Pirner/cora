from fastapi import FastAPI
import uvicorn

from src.api.payloads import BaseGeneratePL, MessagePL
from src.llm.models.utils import LLMModelUtils
from src.llm.models.TransformerModel import TransformerModel


app = FastAPI()
model_configs = LLMModelUtils.read_all_llm_configs(config_directory=r'./.llm_configs')
def_config = list(filter(lambda x: 'qwen3_0_6B' == x.model_id, model_configs))[0]
print('[INFO] found {} model configs.'.format(len(model_configs)))
model = TransformerModel(config=def_config)


@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get('/query_models')
async def get_available():
    return {'models': model_configs}

@app.post('/base_generate')
async def base_generate(pl: BaseGeneratePL):
    """
    base generate, so simply text that is being fead through the model and then getting a response
    :param pl: payload that defines the query
    :return:
    """
    if not model.loaded:
        model.load_model()
    outputs = model.generate(text=pl.text)
    return {
        'message': 'in construction',
        "outputs": outputs,
    }

@app.post('/message_generate')
async def message_generate(pl: MessagePL):
    """
    process messages given by the system
    :param pl:
    :return:
    """
    if not model.loaded:
        model.load_model()
    outputs = model.process_messages(pl=pl)
    return {
        'message': 'in construction',
        "outputs": outputs,
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

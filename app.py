from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from src.api.payloads import BaseGeneratePL, MessagePL, StructuredOutputPL, ChatMessage
from src.llm.models.utils import LLMModelUtils
from src.llm.models.TransformerModel import TransformerModel


app = FastAPI()

# Serve static files (optional for CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Templates location
templates = Jinja2Templates(directory="templates")

model_configs = LLMModelUtils.read_all_llm_configs(config_directory=r'./.llm_configs')
def_config = list(filter(lambda x: 'qwen3_0_6B' == x.model_id, model_configs))[0]
print('[INFO] found {} model configs.'.format(len(model_configs)))
model = TransformerModel(config=def_config)


# @app.get("/")
# async def root():
# return {"message": "Hello World"}

# Show the HTML page at "/"
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})


@app.post("/chat")
async def chat_with_agent(msg: ChatMessage):
    user_input = msg.message
    # Replace this with your actual agent logic
    import time
    time.sleep(5)
    agent_response = f"You said: {user_input}"
    return {"response": agent_response}


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

@app.post('/generate_structured_output')
async def generate_structured_output(pl: StructuredOutputPL):
    print('[Debug]', pl)
    if not model.loaded:
        model.load_model()
    outputs = model.process_structured_output(pl=pl)
    return {
        'message': 'in construction',
        'outputs': outputs,
    }

@app.post('/message_generate')
async def message_generate(pl: MessagePL):
    """
    process messages given by the system, this allows for memory and tool calling.
    This should be the default go to for processing messages through the llm.
    :param pl: payload for jinja style.
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

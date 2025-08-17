from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn

from src.api.payloads import BaseGeneratePL, MessagePL, StructuredOutputPL, ChatHistory
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
async def chat_endpoint(chat: ChatHistory):
    # chat.history is the full list of previous messages
    # You can build your context here, e.g., concatenate or process them
    # context_text = "\n".join(f"{msg.sender}: {msg.text}" for msg in chat.history)
    # print(chat)
    # For demo: just echo back last user message reversed or a static reply
    # user_messages = [msg.text for msg in chat.history if msg.sender == 'user']
    # last_user_message = user_messages[-1] if user_messages else ""
    # user_chats = list(filter(lambda x: x.sender == 'user', chat.history))
    # Your agent logic here, e.g., call an AI with context_text
    # response_text = f"Echo: {user_chats[-1].text}"
    if not model.loaded:
        model.load_model()
    response_text = model.process_chat(chat=chat)

    return {"response": response_text}


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

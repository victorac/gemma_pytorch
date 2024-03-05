from fastapi import FastAPI
from pydantic import BaseModel


class Sentence(BaseModel):
    content: str
    length: int

import contextlib
import random

import numpy as np
import torch

from gemma import config
from gemma import model as gemma_model


CKPT_PATH = "/tmp/ckpt/gemma-2b-it.ckpt"
DEVICE = "cuda"
VARIANT = "2b"
DTYPE = "float16" if DEVICE == "cuda" else "float32"
SEED = 12345

@contextlib.contextmanager
def _set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)

def load_model():
    model_config = config.get_model_config(VARIANT)
    model_config.dtype = DTYPE

    # Seed random.
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Create the model and load the weights.
    device = torch.device(DEVICE)
    with _set_default_tensor_type(model_config.get_dtype()):
        model = gemma_model.GemmaForCausalLM(model_config)
        model.load_weights(CKPT_PATH)
        model = model.to(device).eval()
    print("Model loading done")
    return model, device

def run_prompt(model, prompt, device, output_len):
    result = model.generate(prompt, device, output_len=output_len)

    # Print the prompts and results.
    print('======================================')
    print(f'PROMPT: {prompt}')
    print(f'RESULT: {result}')
    print('======================================')
    return result


MODEL, DEVICE = load_model()
app = FastAPI()

@app.get("/")
def read_root():
    prompt = "Can you help improve my writing? How should I prompt you sentences that I want to be improved?"
    result = run_prompt(model=MODEL, prompt=prompt, device=DEVICE, output_len=len(prompt) + 100)
    return {"result": result}

@app.post("/")
async def submit_prompt(sentence: Sentence):
    prompt = f'''Please proofread the following sentence and suggest improvements for clarity, conciseness, and overall ease of reading. Focus on simplifying complex sentence structures, removing unnecessary words, and ensuring the sentence flows smoothly.
    {sentence.content}
    '''
    result = run_prompt(model=MODEL, prompt=prompt, device=DEVICE, output_len=len(prompt) + sentence.length)
    return {"result": result}


#!/usr/bin/env python
# coding: utf-8

# In[13]:


import os
#MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"
#MODEL_NAME = "Qwen/Qwen3-VL-4B-Instruct"
MODEL_NAME = "Qwen/Qwen3-VL-2B-Instruct"
# Run before import HF transformers.
os.environ["HF_HOME"] = "hf_cache"

import asyncio
import threading
from typing import Optional

from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
from accelerate import Accelerator
from PIL import Image
import requests
from io import BytesIO

from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
)

# Service name is required for most backends
resource = Resource.create(attributes={
    SERVICE_NAME: __name__
})

tracerProvider = TracerProvider(resource=resource)
JAEGER_URL = os.getenv("JAEGER_URL") or "http://jaeger:4318"
processor = BatchSpanProcessor(OTLPSpanExporter(endpoint=f"{JAEGER_URL}/v1/traces"))
tracerProvider.add_span_processor(processor)
trace.set_tracer_provider(tracerProvider)
tracer = trace.get_tracer(__name__)

processor: AutoProcessor
model: AutoModelForImageTextToText = None
accelerator: Accelerator

@asynccontextmanager
async def lifespan(app: FastAPI):
    global init_model_thread
    init_model_thread.start()
    yield
    
app = FastAPI(title=__name__, lifespan=lifespan)
instrumentor = FastAPIInstrumentor()
instrumentor.instrument_app(app)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allowed origins
    allow_credentials=True,         # Allow cookies/auth headers
    allow_methods=["*"],            # Allow all HTTP methods
    allow_headers=["*"],            # Allow all HTTP headers
)

@app.post("/generate")
async def generate(
    image: UploadFile,
    question: str = Form(...),
):
    global tracer, model
    if model == None:
        raise HTTPException(
            status_code=400,
            detail=[{"msg": "VLM is not ready. Check the backend log for keyword 'init_model done.'"}]
        )

    with tracer.start_as_current_span("generate"):
        #image_bytes = BytesIO(requests.get("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG").content)
        image_bytes = await image.read()
        answer = ask_model(image_bytes, question)
        return {"answer": answer}

def init_model():
    print("init_model start...", flush=True)
    global MODEL_NAME, processor, model, accelerator
    # Initialize accelerator
    accelerator = Accelerator()

    min_pixels = 256*28*28
    max_pixels = 2560*28*28
    # Load model and processor
    processor = AutoProcessor.from_pretrained(MODEL_NAME, min_pixels=min_pixels, max_pixels=max_pixels)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    print("init_model done!", flush=True)

init_model_thread = threading.Thread(target=init_model)

def ask_model(
    image_bytes,
    question: str = "What is the animal on the candy?"):
    global tracer, MODEL_NAME, processor, model, accelerator
    image = Image.open(BytesIO(image_bytes)).convert("RGB")

    #print(question)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question}
            ],
        }
    ]

    # Apply chat template to prepare input for the model
    text_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    # Process inputs
    inputs = processor(text=text_prompt, images=image, return_tensors="pt").to(accelerator.device)

    # Generate output
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=256)

    # Decode the generated text
    response = processor.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    #print(response)
    return response

@app.get("/")
async def index():
    if model == None:
        return "Not ready!"
    return "Hi."


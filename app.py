import json
import os
import gradio

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Any

MODELS = [
    "ibm-granite/granite-embedding-30m-english",
    "ibm-granite/granite-embedding-278m-multilingual"
]

current_model = None
model = None
app = FastAPI()

def load_model(model_name: str):
    global current_model

    if current_model is not None and current_model == model_name:
        return current_model
    
    try:
        current_model = SentenceTransformer(model_name)
    except Exception as ex:
        raise ValueError(f"Failed to load model '{model_name}': {str(ex)}")

    return current_model

def embed(document: str, model_name: str):
    if model_name:
        try:
            new_model = load_model(model_name)
            return new_model.encode(document)
        except Exception as ex:
            raise ValueError(f"Failed to load model '{model_name}': {str(ex)}")
    
    return None

@app.get("/models")
async def get_models():
    return JSONResponse(
        content={
            "models": MODELS
        }
    )

@app.post("/embed")
async def generate_embedding(data: Dict[str, Any]):
    try:
        text = data.get("text", "")
        model_name = data.get("model","")

        if not text:
            return JSONResponse(
                status_code=400,
                content={"error": "No text provided"}
            )
        
        if model_name not in MODELS:
            message = f"Only IBM Granite embedding models can be used: {MODELS}"
            return JSONResponse(
                status_code=400,
                content={"error": message}
            )
        
        if model_name:
            vector_embedding = embed(text, model_name)
        
            return JSONResponse(
                content={
                    "embedding": vector_embedding.tolist(),
                    "dim": len(vector_embedding),
                    "model": model_name
                }
        )
        
    except Exception as ex:
        return JSONResponse(
            status_code=500,
            content={"error": str(ex)}
        )

with gradio.Blocks(title="Multi-Model Text Embeddings", css_path="./style.css") as gradio_app:
    gradio.Markdown("# Multi-Model Text Embeddings")
    gradio.Markdown("Generate embeddings for your text using the IBM Granite embedding models.")
    
    # Model selector dropdown (allows custom input)
    model_dropdown = gradio.Dropdown(
        choices=MODELS,
        value="",
        label="Select Embedding Model",
        info="Choose any predefined model name",
        allow_custom_value=True
    )
    
    # Create an input text box
    text_input = gradio.Textbox(label="Enter text to embed", placeholder="Type or paste your text here...")

    # Create an output component to display the embedding
    output = gradio.JSON(label="Text Embedding", elem_classes=["json-holder"])
    
    # Add a submit button with API name
    submit_btn = gradio.Button("Generate Embedding", variant="primary")

    # Handle both button click and text submission
    submit_btn.click(embed, inputs=[text_input, model_dropdown], outputs=output, api_name="predict")
    text_input.submit(embed, inputs=[text_input, model_dropdown], outputs=output)
    
    # Add API usage guide
    gradio.Markdown("## API Usage")
    gradio.Markdown("""
    You can use this API in two ways: via the direct FastAPI endpoint or through Gradio clients.
    
    ### List Available Models
    ```bash
    curl https://aploetz-granite-embeddings.hf.space/models
    ```
    
    ### Direct API Endpoint (No Queue!)
    ```bash
    # Default model (nomic-ai/nomic-embed-text-v1.5)
    curl -X POST https://ipepe-nomic-embeddings.hf.space/embed \
      -H "Content-Type: application/json" \
      -d '{"text": "Your text to embed goes here"}'
    
    # With predefined model (trust_remote_code allowed)
    curl -X POST https://ipepe-nomic-embeddings.hf.space/embed \
      -H "Content-Type: application/json" \
      -d '{"text": "Your text to embed goes here", "model": "sentence-transformers/all-MiniLM-L6-v2"}'
    
    # With any Hugging Face model (trust_remote_code=False for security)
    curl -X POST https://ipepe-nomic-embeddings.hf.space/embed \
      -H "Content-Type: application/json" \
      -d '{"text": "Your text to embed goes here", "model": "intfloat/e5-base-v2"}'
    ```
    
    Response format:
    ```json
    {
      "embedding": [0.123, -0.456, ...],
      "dim": 384,
      "model": "sentence-transformers/all-MiniLM-L6-v2",
      "trust_remote_code": false,
      "predefined": true
    }
    ```
    
    ### Python Example (Direct API)
    ```python
    import requests
    
    # List available models
    models = requests.get("https://ipepe-nomic-embeddings.hf.space/models").json()
    print(models["models"])
    
    # Generate embedding with specific model
    response = requests.post(
        "https://ipepe-nomic-embeddings.hf.space/embed",
        json={
            "text": "Your text to embed goes here",
            "model": "BAAI/bge-small-en-v1.5"
        }
    )
    result = response.json()
    embedding = result["embedding"]
    ```
    
    ### Python Example (Gradio Client)
    ```python
    from gradio_client import Client
    
    client = Client("ipepe/nomic-embeddings")
    result = client.predict(
        "Your text to embed goes here",
        "nomic-ai/nomic-embed-text-v1.5",  # model selection
        api_name="/predict"
    )
    print(result)  # Returns the embedding array
    ```
    
    ### Available Models
    - `ibm-granite/granite-embedding-30m-english` - IBM Granite 30M English embedding model
    - `ibm-granite/granite-embedding-278m-multilingual` - IBM Granite 278M multilingual embedding model
    """)

if __name__ == '__main__':
    # Mount FastAPI app to Gradio
    gradio_app = gradio.mount_gradio_app(app, gradio_app, path="/")
    
    # Run with Uvicorn (Gradio uses this internally)
    import uvicorn
    uvicorn.run(gradio_app, host="0.0.0.0", port=7860)
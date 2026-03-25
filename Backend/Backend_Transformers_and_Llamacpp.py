from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from llama_cpp import Llama
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
from huggingface_hub import hf_hub_download
from threading import Thread
import torch

app = Flask(__name__)
CORS(app)

# Transformer compatibility services
Processorwork = True
Cuda = False
Rocm = False

# Model
ModelUsed = "Qwen/Qwen3-0.6B"
AvalaibleModel = [
    {"name": "Qwen3-0.6B", "Provider": "AlibabaCloud", "Parameter": "0.6B", "ram": "2GB"},
    {"name": "Qwen3-4B",   "Provider": "AlibabaCloud", "Parameter": "4B",   "ram": "8GB"},
]

AI_Role = "You are a helpful assistant."
Model_Format = "gguf"

# Modele parameter
if Model_Format == "gguf":
    MODEL_PATH = hf_hub_download(
        repo_id="ggml-org/Qwen3-4B-GGUF",
        filename="Qwen3-4B-Q4_K_M.gguf",
        local_dir="./models"
    )
    
    model = Llama(model_path=MODEL_PATH, n_ctx=2048)
    tokenizer = None
else:
    tokenizer = AutoTokenizer.from_pretrained(ModelUsed)
    model = AutoModelForCausalLM.from_pretrained(ModelUsed, torch_dtype=torch.bfloat16)

conversation = [{"role": "system", "content": AI_Role}]

#token generation
@app.route("/API/Streamer", methods=["POST"])
def streaming_Service():
    global conversation
    asked_question = request.json["message"]
    conversation.append({"role": "user", "content": asked_question})

    if Model_Format == "gguf":
        def generer():
            reponse_complete = ""
            for chunk in model.create_chat_completion(
                messages=conversation,
                max_tokens=700,
                temperature=0.7,
                stream=True
            ):
                token = chunk["choices"][0]["delta"].get("content", "")
                reponse_complete += token
                yield f"{token}\n\n"
            conversation.append({"role": "assistant", "content": reponse_complete})

    else:
        input_text = tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )
        tokenized_input = tokenizer(input_text, return_tensors="pt")
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

        thread = Thread(target=model.generate, kwargs={
            **tokenized_input,
            "max_new_tokens": 700,
            "temperature": 0.7,
            "do_sample": True,
            "streamer": streamer
        })
        thread.start()

        def generer():
            reponse_complete = ""
            for token in streamer:
                reponse_complete += token
                yield f"{token}\n\n"
            conversation.append({"role": "assistant", "content": reponse_complete})

    return Response(generer(), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(port=5000, debug=False)
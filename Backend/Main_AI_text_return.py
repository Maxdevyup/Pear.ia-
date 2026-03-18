from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM, TextIteratorStreamer
import torch
from threading import Thread

app = Flask(__name__)
CORS(app)
#Gpu type
Processorwork = True
Cuda = False
Rocm = False

#Model
ModelUsed = "Qwen/Qwen3-0.6B" #Place holder for testing, get a small Ai fast like this one ( 0.6b to 4b for fast response)
AvalaibleModel = [
    {"name": "Qwen3-0.6B", "Provider": "AlibabaCloud", "Parameter": "0.6B", "ram": "2GB"},
    {"name": "Qwen3-4B",  "Provider": "AlibabaCloud", "Parameter": "4B",   "ram": "8GB"},

]

#Role
AI_Role = "You are a helpful assistant." #Place Holder
#Context
conversation = [
    {"role": "system", "content": AI_Role}
]


#Tokenizer type
model_id = ModelUsed
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16)


#Generating message func
@app.route("/API/Streamer", methods=["POST"])
def Stream_Response():
    global conversation
    asked_question = request.json["message"]
    conversation.append({"role": "user", "content": asked_question})

    Input_text = tokenizer.apply_chat_template(
    conversation,
     tokenize=False,
        add_generation_prompt=True
    )
    TokenizeInput = tokenizer(Input_text, return_tensors="pt")
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False, enable_thinking = False)

    thread = Thread(target=model.generate, kwargs={
        **TokenizeInput,
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
            yield f" {token}\n\n"
        conversation.append({"role": "assistant", "content": reponse_complete})

    return Response(generer(), mimetype="text/event-stream")


#PY Server 
if __name__ == "__main__":
    app.run(port=5000, debug=False)
    
        
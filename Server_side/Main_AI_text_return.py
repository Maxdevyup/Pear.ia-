
from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)
CORS(app)


#Token management
TokenAvailable = 100
TokenUsed = 0

# Load the model and tokenizer
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)


@app.route('/get_response', methods=['POST'])
def response_send():
    #question input
    asked_question = request.json["message"]
    #Create the message
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": asked_question}
    ]

    #Create the input for the model
    Input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    TokenizeInput = tokenizer(Input_text, return_tensors="pt")

    #Generate the response
    with torch.no_grad():
        outputs = model.generate(
        **TokenizeInput,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(response)
    return jsonify({"reponse": response}), print("hello world")

    #Launch Server
if __name__ == "__main__":
    app.run(port=5000, debug=False)
    
        
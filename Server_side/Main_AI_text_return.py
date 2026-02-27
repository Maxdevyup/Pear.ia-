
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
asked_question = input("What is your question? ")
TokenAvailable = 100
TokenUsed = 0
model_id = "Qwen/Qwen3-0.6B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": asked_question}
]
Input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
TokenizeInput = tokenizer(Input_text, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **TokenizeInput,
        max_new_tokens=200,
        temperature=0.7,
        do_sample=True
    )
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
"""module for evaluating an mteb task"""
from transformers import AutoTokenizer

pretrained_model_name = "Salesforce/xgen-7b-8k-base"
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name, trust_remote_code=True)

tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer)) # initialize model embeddings with this size
model = prepare_model_for_int8_training(model) # init
training_args = model_training_args

from unsloth import FastLanguageModel, is_bfloat16_supported, train_on_responses_only
from datasets import load_dataset, Dataset

from trl import SFTTrainer, apply_chat_template

from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer

import torch

#Model
max_seq_length = 2048 
dtype = None # None for auto-detection.
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
	model_name = "unsloth/Llama-3.2-3B-Instruct",
	max_seq_length = max_seq_length,
	dtype = dtype,
	load_in_4bit = load_in_4bit,
	# token = "hf_...", # use if using gated models like meta-llama/Llama-3.2-11b
)

#Initialize
model = FastLanguageModel.get_peft_model(
	model,
	r = 16,
	target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                  	"gate_proj", "up_proj", "down_proj",],
	lora_alpha = 16,
	lora_dropout = 0, 
	bias = "none",
	use_gradient_checkpointing = "unsloth",
	random_state = 42,
	use_rslora = False, 
	loftq_config = None,
)

#Data Processing
dataset = load_dataset("APIData.xlsx", split = "train")

def convert_dataset_to_dict(dataset):
    dataset_dict = {
        "prompt": [],
        "completion": []
    }

    for row in dataset:
        user_content = f"Context: {row['context']}\nQuestion: {row['question']}"
        assistant_content = row['answer']

        dataset_dict["prompt"].append([
            {"role": "user", "content": user_content}
        ])
        dataset_dict["completion"].append([
            {"role": "assistant", "content": assistant_content}
        ])
    return dataset_dict
    
    
converted_data = convert_dataset_to_dict(dataset)
dataset = Dataset.from_dict(converted_data)
dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

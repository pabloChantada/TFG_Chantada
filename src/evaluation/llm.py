import json
import os
from huggingface_hub import login
import torch
from PIL import Image
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from os import getenv
    
login(token=os.getenv("HUGGINGFACE_TOKEN"))

MODEL_ID = "openbmb/MiniCPM-V-2_6"

processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
tokenizer = processor.tokenizer

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    device_map="auto"
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
model = get_peft_model(model, lora_config)

def make_prompt(state):
    state_str = json.dumps(state, ensure_ascii=False)
    # Formato simple con token de imagen
    return f"<image>\nEstado: {state_str}\nAcción:"

def preprocess(example):
    image = Image.open(example["image"]).convert("RGB")
    prompt = make_prompt(example["state"])
    action = example["action"]

    # Encode prompt + image
    inputs = processor(images=image, text=prompt, return_tensors="pt")

    prompt_ids = inputs["input_ids"][0]
    action_ids = tokenizer(action, add_special_tokens=False).input_ids

    input_ids = torch.cat([prompt_ids, torch.tensor(action_ids)], dim=0)
    attention_mask = torch.ones_like(input_ids)
    labels = torch.cat([torch.full((len(prompt_ids),), -100), torch.tensor(action_ids)], dim=0)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": inputs["pixel_values"][0]
    }

def collate_fn(batch):
    input_ids = [b["input_ids"] for b in batch]
    attention_mask = [b["attention_mask"] for b in batch]
    labels = [b["labels"] for b in batch]
    pixel_values = torch.stack([b["pixel_values"] for b in batch])

    input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
    labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "pixel_values": pixel_values
    }

dataset = load_dataset("json", data_files={"train": "data/train.jsonl"})
train_dataset = dataset["train"].map(preprocess, remove_columns=dataset["train"].column_names)

args = TrainingArguments(
    output_dir="out_minicpmv",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=3,
    fp16=True,
    learning_rate=2e-4,
    logging_steps=20,
    save_steps=200,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    data_collator=collate_fn
)

trainer.train()
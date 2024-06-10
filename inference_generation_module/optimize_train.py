import tqdm
from datasets import load_dataset
import csv
# import lawrouge

# import datasets
import random
import pandas as pd

# from datasets import dataset_dict
import datasets

from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import AutoTokenizer, BertConfig
import warnings
from pathlib import Path
from typing import List, Tuple, Union

# from torch import nn

import numpy as np
import lawrouge
import torch

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel
from transformers.utils import logging
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
dataset = load_dataset('csv', data_files='./generate_data/sentence_inference.csv')


TokenModel = "facebook/bart-base"

# from transformers import AutoTokenizer, BertConfig

tokenizer = AutoTokenizer.from_pretrained(TokenModel)

# config = BertConfig.from_pretrained(TokenModel)

model_checkpoint = "facebook/bart-base"

print('model_checkpoint: ', model_checkpoint)

max_input_length = 512
max_target_length = 128


def preprocess_function(examples):
    input_section = [doc for doc in examples['sentence']]
    model_inputs = tokenizer(input_section, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["inference"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_data_txt, validation_data_txt = dataset['train'].train_test_split(test_size=0.1, shuffle=True, seed=42).values()
dd = datasets.DatasetDict({"train": train_data_txt, "val": validation_data_txt})

raw_datasets = dd
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)

logger = logging.get_logger(__name__)

batch_size = 128  # 4
args = Seq2SeqTrainingArguments(
    output_dir="sentence_optimize_1",
    num_train_epochs=50,  # 50,  # demo
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=batch_size,  # demo
    per_device_eval_batch_size=batch_size,
    learning_rate=1e-04,
    warmup_steps=500,
    weight_decay=0.001,
    label_smoothing_factor=0.1,
    predict_with_generate=True,
    logging_dir="logs",
    logging_steps=1000,
    evaluation_strategy="epoch",
    save_total_limit=3,
    generation_max_length=64,
    generation_num_beams=3
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)


trainer = Seq2SeqTrainer(
    model,
    args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["val"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    # compute_metrics=compute_metrics
)


model.load_state_dict(torch.load('./sentence_optimize_1/pytorch_model.bin'))
model = AutoModelForSeq2SeqLM.from_pretrained('./sentence_optimize_1/checkpoint-2500')
tokenizer = AutoTokenizer.from_pretrained('./sentence_optimize_1/checkpoint-2500')
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# print(device)
# print(model.device)


def generate_question(test_samples, model):
    inputs = tokenizer(
        test_samples,
        padding="max_length",
        truncation=True,
        # max_length=max_input_length,
        max_length=512,
        return_tensors="pt",
    )
    # print('inputs', inputs)
    model = model.to(device)
    input_ids = inputs.input_ids.to(model.device)

    attention_mask = inputs.attention_mask.to(model.device)

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
    # print('outputs: ', outputs)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


f = open(r'./generate_data/sentence_optimize_1.csv', 'w', encoding='utf-8', newline='')
r1 = csv.writer(f)
header = ['optimize_sentence']
r1.writerow(header)
dataset = load_dataset('csv', data_files='./generate_data/section_importantSentence.csv')
sentences = dataset['train']['important_sentence']
num = len(sentences)
for n in tqdm.tqdm(range(num)):
    _, x = generate_question(sentences[n], model)
    raw = [x[0]]
    r1.writerow(raw)
    # print(x[0])
f.close()

# # f = open(r'./generate_data/sentence_optimize3.csv', 'w', encoding='utf-8', newline='')
# # r1 = csv.writer(f)
# # header = ['optimize_sentence']
# # r1.writerow(header)
# # dataset = load_dataset('csv', data_files='./generate_data/section_importantSentence1.csv')
# # sentences = dataset['train']['important_sentence']
# # num = len(sentences)
# # for n in tqdm.tqdm(range(num)):
# #     _, x = generate_question(sentences[n], model)
# #     raw = [x[0]]
# #     r1.writerow(raw)
# #     # print(x[0])
# # f.close()

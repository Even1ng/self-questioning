import tqdm
from datasets import load_dataset
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
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

data_files = {'train': './data/train_qa.csv', 'test': './data/test_qa.csv', 'val': './data/val_qa.csv'}
dataset = load_dataset('csv', data_files=data_files)
TokenModel = "facebook/bart-base"
tokenizer = AutoTokenizer.from_pretrained(TokenModel)
model_checkpoint = "facebook/bart-base"

print('model_checkpoint: ', model_checkpoint)

max_input_length = 512
max_target_length = 128


def preprocess_function(examples):
    input_section = [doc for doc in examples['cor_section']]
    input_question = [inf for inf in examples['question']]
    inputs = [str(question) + str(section) for question, section in zip(input_question, input_section)]
    model_inputs = tokenizer(inputs, max_length=max_input_length, truncation=True)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["answer"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


raw_datasets = dataset
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
logger = logging.get_logger(__name__)

batch_size = 128  # 4
args = Seq2SeqTrainingArguments(
    output_dir="answer_results3",
    num_train_epochs=10,  # 50,  # demo
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

train_result = trainer.train()
trainer.save_model()
trainer.save_state()



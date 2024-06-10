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
model_checkpoint = "facebook/bart-base"
max_input_length = 512
max_target_length = 128


model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)


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
header = ['section', 'optimize_sentence']
r1.writerow(header)
dataset = load_dataset('csv', data_files='./generate_data/section_importantSentence.csv')
sentences = dataset['train']['important_sentence']
sections = dataset['train']['section']
num = len(sentences)
for n in tqdm.tqdm(range(num)):
    _, x = generate_question(sentences[n], model)
    raw = [sections[n], x[0]]
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

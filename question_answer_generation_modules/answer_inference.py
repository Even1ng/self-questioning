from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from transformers import BertConfig
import torch
import tqdm
"""gt question + section → answer"""

model_checkpoint = "facebook/bart-base"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.load_state_dict(torch.load('./answer_results3/checkpoint-23000/pytorch_model.bin'))
model = AutoModelForSeq2SeqLM.from_pretrained('./answer_results3/checkpoint-23000')
tokenizer = AutoTokenizer.from_pretrained('./answer_results3/checkpoint-23000')
max_input_length = 512


def generate_answer(contents, model):
    inputs = tokenizer(
        contents,
        padding="max_length",
        truncation=True,
        # max_length=max_input_length,
        max_length=512,
        return_tensors="pt",
    )
    # print('inputs', inputs)
    input_ids = inputs.input_ids.to(model.device)

    attention_mask = inputs.attention_mask.to(model.device)

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=128)
    # print('outputs: ', outputs)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return output_str


dataset = load_dataset('csv', data_files='./data/test_qa.csv')
questions = dataset['train']['question']
sections = dataset['train']['cor_section']
with open('answer_eval.txt', 'w', encoding='utf-8') as f:
    num = len(questions)
    for n in tqdm.tqdm(range(num)):
        inputs_content = questions[n] + sections[n]
        x = generate_answer(inputs_content, model)
        f.write(x[0] + '\n')
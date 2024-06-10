from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSeq2SeqLM
from transformers import BertConfig
import torch
import tqdm


n_tokens = 20  # 512
model_checkpoint = "facebook/bart-large"
model = AutoModelForSeq2SeqLM.from_pretrained(model_checkpoint)
model.load_state_dict(torch.load('./question_filter5/checkpoint-1000/pytorch_model.bin'))
tokenizer = AutoTokenizer.from_pretrained('./question_filter5/checkpoint-1000')
device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
# # print(model.device)


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
    inputs['input_ids'] = torch.cat([torch.full((1, n_tokens), 50256), inputs['input_ids']], 1)
    inputs['attention_mask'] = torch.cat([torch.full((1, n_tokens), 1), inputs['attention_mask']], 1)

    model = model.to(device)
    input_ids = inputs.input_ids.to(model.device)

    attention_mask = inputs.attention_mask.to(model.device)

    outputs = model.generate(input_ids, attention_mask=attention_mask, max_length=64)
    # print('outputs: ', outputs)
    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return outputs, output_str


dataset = load_dataset('csv', data_files='./data/sentence_optimize_1.csv')
sentences = dataset['train']['important_sentence']
sections = dataset['train']['section']
with open('question.txt', 'w', encoding='utf-8') as f:
    num = len(sentences)
    for n in tqdm.tqdm(range(num)):
        _, x = generate_question(sentences[n] + sections[n], model)
        f.write(x[0] + '\n')
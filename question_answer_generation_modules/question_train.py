import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import tqdm
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
# from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from torch import nn
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, PreTrainedModel, BartForConditionalGeneration
from transformers.utils import logging


'''
以inference和section作为输入，输出是question
encoder前加了soft prompt
对filter5 generate_question()的改进
'''


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')

data_files = {'train': './data/train_qa.csv', 'test': './data/test_qa.csv', 'val': './data/val_qa.csv'}
dataset = load_dataset('csv', data_files=data_files)
TokenModel = "facebook/bart-large"
tokenizer = AutoTokenizer.from_pretrained(TokenModel)
model_checkpoint = "facebook/bart-large"
print('model_checkpoint: ', model_checkpoint)

max_input_length = 512
max_target_length = 64

n_tokens = 20  # 512


def preprocess_function(examples):
    input_section = [doc for doc in examples['cor_section']]
    input_inference = [inf for inf in examples['inference']]
    inputs = [str(inference) + str(section) for inference, section in zip(input_inference, input_section)]
    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding='max_length', return_tensors="pt")
    # print(model_inputs['input_ids'].shape)  # batch size=1000 for map function
    model_inputs['input_ids'] = torch.cat([torch.full((1000, n_tokens), 50256), model_inputs['input_ids']], 1)
    model_inputs['attention_mask'] = torch.cat([torch.full((1000, n_tokens), 1), model_inputs['attention_mask']], 1)

    # Set up the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["question"], max_length=max_target_length, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


raw_datasets = dataset
tokenized_datasets = raw_datasets.map(preprocess_function, batched=True, drop_last_batch=True)
model = BartForConditionalGeneration.from_pretrained(model_checkpoint)


class SoftEmbedding(nn.Module):
    def __init__(self,
                 wte: nn.Embedding,
                 n_tokens: int = 10,
                 random_range: float = 0.5,
                 initialize_from_vocab: bool = True):
        """appends learned embedding to
        Args:
            wte (nn.Embedding): original transformer word embedding
            n_tokens (int, optional): number of tokens for task. Defaults to 10.
            random_range (float, optional): range to init embedding (if not initialize from vocab). Defaults to 0.5.
            initialize_from_vocab (bool, optional): initalizes from default vocab. Defaults to True.
        """
        super(SoftEmbedding, self).__init__()
        self.wte = wte
        self.n_tokens = n_tokens
        self.learned_embedding = nn.Parameter(self.initialize_embedding(wte, n_tokens, random_range, initialize_from_vocab))

    def initialize_embedding(self,
                             wte: nn.Embedding,
                             n_tokens: int = 10,
                             random_range: float = 0.5,
                             initialize_from_vocab: bool = True):
        """initializes learned embedding
        Args:
            same as __init__
        Returns:
            torch.float: initialized using original schemes
        """
        if initialize_from_vocab:
            return self.wte.weight[:n_tokens].clone().detach()
        return torch.FloatTensor(n_tokens, wte.weight.size(1)).uniform_(-random_range, random_range)

    def forward(self, tokens):
        """run forward pass
        Args:
            tokens (torch.long): input tokens before encoding
        Returns:
            torch.float: encoding of text concatenated with learned task specifc embedding
        """
        input_embedding = self.wte(tokens[:, self.n_tokens:])
        learned_embedding = self.learned_embedding.repeat(input_embedding.size(0), 1, 1)
        return torch.cat([learned_embedding, input_embedding], 1)


initialize_from_vocab = True
s_wte = SoftEmbedding(model.get_input_embeddings(), n_tokens=n_tokens, initialize_from_vocab=initialize_from_vocab)
model.set_input_embeddings(s_wte)
logger = logging.get_logger(__name__)

batch_size = 64  # 4
args = Seq2SeqTrainingArguments(
    output_dir="question_filter5",
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
    logging_steps=100,  # 1000
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

# import torch

model.load_state_dict(torch.load('./question_filter5/checkpoint-1000/pytorch_model.bin'))
# for param in model.state_dict():
#     print(param)
# print(model)
# model = AutoModelForSeq2SeqLM.from_pretrained('./question_filter5/checkpoint-1000')
tokenizer = AutoTokenizer.from_pretrained('./question_filter5/checkpoint-1000')
# model.load_state_dict(torch.load('./question_filter2_test/pytorch_model.bin'))
# model = AutoModelForSeq2SeqLM.from_pretrained('./question_filter2_test')
# tokenizer = AutoTokenizer.from_pretrained('./question_filter2_test')
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


dataset = load_dataset('csv', data_files='./data/section_importantSentence.csv')
sentences = dataset['train']['important_sentence']
sections = dataset['train']['cor_section']
with open('question_without_bart1.txt', 'w', encoding='utf-8') as f:
    num = len(sentences)
    for n in tqdm.tqdm(range(num)):
        # print(f'已完成{n}/{num}')
        _, x = generate_question(sentences[n] + sections[n], model)
        f.write(x[0] + '\n')

# dataset = load_dataset('csv', data_files='./data/test_qa.csv')
# sentences = dataset['train']['inference']
# sections = dataset['train']['cor_section']
# num = len(sentences)
# for n in tqdm.tqdm(range(num)):
#     _, x = generate_question(sentences[n] + sections[n], model)
#     print(x[0])
# '''用inference代替important sentence，质量明显升高'''

# dataset = load_dataset('csv', data_files='./data/test_qa.csv')
# sentences = dataset['train']['inference']
# sections = dataset['train']['cor_section']
# with open('question_eval3_1.txt', 'w', encoding='utf-8') as f:
#     num = len(sentences)
#     for n in tqdm.tqdm(range(num)):
#         # print(f'已完成{n}/{num}')
#         _, x = generate_question(sentences[n] + sections[n], model)
#         f.write(x[0] + '\n')

# dataset = load_dataset('csv', data_files='./data/section_importantSentence_new3.csv')
# dataset2 = load_dataset('csv', data_files='./data/sentence_optimize_new3.csv')
# # dataset3 = load_dataset('csv', data_files='./data/sentence_optimize2.csv')
# # sentences = dataset['train']['important_sentence']
# sentences = dataset2['train']['optimize_sentence']
# # sentences = dataset3['train']['optimize_sentence']
# sections = dataset['train']['section']
# num = len(sentences)
# # num = 50  # 测试
# # for n in tqdm.tqdm(range(num)):
# #     _, x = generate_question(sentences[n] + sections[n], model)
# #     print(x[0])
# with open('question_new3_1_new3.txt', 'w', encoding='utf-8') as f:
#     for n in tqdm.tqdm(range(num)):
#         # print(f'已完成{n}/{num}')
#         _, x = generate_question(sentences[n] + sections[n], model)
#         f.write(x[0] + '\n')

# dataset = load_dataset('csv', data_files='./data/section_importantSentence1.csv')
# dataset2 = load_dataset('csv', data_files='./data/sentence_optimize3.csv')
# sentences = dataset2['train']['optimize_sentence']
# sections = dataset['train']['cor_section']
# # num = len(sentences)
# num = 50  # 测试
# for n in tqdm.tqdm(range(num)):
#     _, x = generate_question(sentences[n] + sections[n], model)
#     print(x[0])


# dataset = load_dataset('csv', data_files='./data/score_optimize.csv')
# sentences = dataset['train']['optimize_sentence']
# sections = dataset['train']['section']
# num = len(sentences)
# # num = 50  # 测试
# # for n in tqdm.tqdm(range(num)):
# #     _, x = generate_question(sentences[n] + sections[n], model)
# #     # print(x[0])
# with open('question_new3_1.txt', 'w', encoding='utf-8') as f:
#     for n in tqdm.tqdm(range(num)):
#         # print(f'已完成{n}/{num}')
#         _, x = generate_question(sentences[n] + sections[n], model)
#         f.write(x[0] + '\n')





import torch
import torch.nn as nn
import csv
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import tqdm


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 1).to(device)

    def forward(self, x1, x2):
        sentence_bert = SentenceTransformer('roberta-base-nli-mean-tokens')
        x1 = sentence_bert.encode(x1, convert_to_tensor=True).to(device)
        x2 = sentence_bert.encode(x2, convert_to_tensor=True).to(device)
        out = self.linear(x1 + x2).to(device)
        out = out.squeeze(-1)
        return out


model = LinearRegression()
data = load_dataset('csv', data_files='./clean_data/test_clean.csv')
temp = torch.load('score_roberta.pth')
model.load_state_dict(temp)


def inference(input_sentence, input_section, model):
    model = model.to(device)
    prediction_score = model(input_sentence, input_section)
    return prediction_score


sentence_score = {}
section_importantSentence = {}


input_section = data['train']['cor_section']
n = 1
section_id = 1
book_id = data['train']['book_id'][0]
f = open(r'generate_data/section_importantSentence.csv', 'w', encoding='utf-8', newline='')
r1 = csv.writer(f)
header = ['important_sentence', 'section']
r1.writerow(header)
for n in tqdm.tqdm(range(len(data['train']))):
    if data['train']['section_id'][n] != section_id:
        cor_section = data['train']['cor_section'][n - 1]

        for s in sent_tokenize(cor_section):
            score = inference(s, cor_section, model)
            sentence_score[s] = score.item()

        sentence_score_filter = {sentence: score for sentence, score in sentence_score.items() if score >= 0.6}
        sentences = sorted(sentence_score_filter, reverse=True, key=sentence_score.get)

        for sentence in sentences:
            section_importantSentence[sentence] = cor_section
        section_id += 1

        for sentence, section in section_importantSentence.items():
            raw = [sentence, section]
            r1.writerow(raw)
        section_importantSentence = {}
        sentence_score = {}
    if data['train']['book_id'][n] != book_id:
        book_id = data['train']['book_id'][n]
        section_id = 1
    if data['train']['inference'][n] == data['train']['inference'][-1]:
        cor_section = data['train']['cor_section'][-1]

        for s in sent_tokenize(cor_section):
            score = inference(s, cor_section, model)
            sentence_score[s] = score

        sentence_score_filter = {sentence: score for sentence, score in sentence_score.items() if score >= 0.6}
        sentences = sorted(sentence_score_filter, reverse=True, key=sentence_score.get)

        for sentence in sentences:
            section_importantSentence[sentence] = cor_section
        section_id += 1

        for sentence, section in section_importantSentence.items():
            raw = [sentence, section]
            r1.writerow(raw)
        break
f.close()

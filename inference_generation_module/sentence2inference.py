from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import csv
import tqdm

dataset = load_dataset('csv', data_files='./clean_data/train_clean.csv')
model = SentenceTransformer('roberta-base-nli-mean-tokens')

book_id = dataset['train']['book_id'][0]
inference_location = 0
section_id = 1
sim_dict = {}
sentence_inference_dict = {}

# for n in tqdm.tqdm(range(50)):  # 测试
for n in tqdm.tqdm(range(len(dataset['train']))):
    if dataset['train']['section_id'][n] != section_id:  # 当section_id变化时，代表着一章内容结束
        cor_section = dataset['train']['cor_section'][n-1]
        for i in dataset['train']['inference'][inference_location: n]:
            for s in sent_tokenize(cor_section):
                cosin_sim = util.pytorch_cos_sim(model.encode(i), model.encode(s))
                sim_dict[s] = cosin_sim
            sentence = sorted(sim_dict.items(), key=lambda x: x[1], reverse=True)[0][0]
            sentence_inference_dict[i] = sentence
            # print(sentence_inference_dict)
            sim_dict = {}
        inference_location += n - inference_location
        # print('inference_location', inference_location)
        section_id += 1
    if dataset['train']['book_id'][n] != book_id:
        book_id = dataset['train']['book_id'][n]
        section_id = 1


f = open(r'./generate_data/sentence_inference.csv', 'w', encoding='utf-8', newline='')
r1 = csv.writer(f)
header = ['inference', 'sentence']
r1.writerow(header)
for inference, sentence in sentence_inference_dict.items():
    raw = [inference, sentence]
    r1.writerow(raw)
f.close()
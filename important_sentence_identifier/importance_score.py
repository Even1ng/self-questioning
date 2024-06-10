from sentence_transformers import SentenceTransformer, util
from datasets import load_dataset
from nltk.tokenize import sent_tokenize
import csv
import tqdm


dataset = load_dataset('csv', data_files='./clean_data/train_clean.csv')
model = SentenceTransformer('roberta-base-nli-mean-tokens')


book_id = dataset['train']['book_id'][0]
inference_location = 0
inference_num = 0
section_id = 1
cosin_list = []
score_dict = {}


for n in tqdm.tqdm(range(len(dataset['train']))):
    if dataset['train']['section_id'][n] != section_id:
        cor_section = dataset['train']['cor_section'][n-1]
        for s in sent_tokenize(cor_section):
            for i in dataset['train']['inference'][inference_location: n]:
                cosin_sim = util.pytorch_cos_sim(model.encode(i), model.encode(s))
                cosin_list += cosin_sim
                inference_num += 1

            score_dict[s] = max(cosin_list)
            cosin_list = []
        inference_location += n - inference_location
        section_id += 1
        inference_num = 0

    if dataset['train']['book_id'][n] != book_id:
        book_id = dataset['train']['book_id'][n]
        section_id = 1


f = open(r'./generate_data/sentenceSection_score.csv', 'w', encoding='utf-8', newline='')
r1 = csv.writer(f)
header = ['sentence', 'score', 'cor_section']
r1.writerow(header)
for sentence, score in score_dict.items():
    raw = [sentence, score]
    r1.writerow(raw)
f.close()
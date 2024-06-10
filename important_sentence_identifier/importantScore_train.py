import torch
import torch.nn as nn
import datasets
from datasets import load_dataset
from sentence_transformers import SentenceTransformer, util
import random


device = torch.device('cuda' if torch.cuda.is_available else 'cpu')
print(device)


def data_generate(input_sentence, input_section, output, batch_size=128, shuffle=True):
    if shuffle:
        data = list(zip(input_sentence, input_section, output))
        random.shuffle(data)
        input_data1, input_data2, output_data = zip(*data)
    else:
        data = list(zip(input_sentence, input_section, output))
        input_data1, input_data2, output_data = zip(*data)
    for i in range(0, len(output_data), batch_size):
        batch_input_sentence = []
        batch_input_section = []
        for line in input_data1[i: i + batch_size]:
            batch_input_sentence.append(line)
        for line_2 in input_data2[i: i + batch_size]:
            batch_input_section.append(line_2)
        batch_output = [label for label in output_data[i: i + batch_size]]
        yield batch_input_sentence, batch_input_section, batch_output


class LinearRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 1).to(device)  # 训练的Sentence_BERT模型得到了固定长度为768的句子表示

    def forward(self, x1, x2):
        sentence_bert = SentenceTransformer('roberta-base-nli-mean-tokens')
        x1 = sentence_bert.encode(x1, convert_to_tensor=True).to(device)
        x2 = sentence_bert.encode(x2, convert_to_tensor=True).to(device)
        out = self.linear(x1 + x2).to(device)
        out = out.squeeze(-1)
        return out


class Linear_Model():
    def __init__(self):
        """
        Initialize the Linear Model
        """
        self.learning_rate = 0.001
        self.epoches = 3
        self.loss_function = nn.MSELoss().to(device)
        self.create_model()

    def create_model(self):
        self.model = LinearRegression()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, data, model_save_path="score_roberta.pth"):
        for epoch in range(self.epoches):
            for sentence, section, score in data_generate(data['train']["sentence"],
                                                          data['train']["cor_section"], data['train']["score"]):
                prediction = self.model(sentence, section).to(device)
                score = torch.tensor(score).to(device)
                # prediction = torch.tensor(prediction)
                loss = self.loss_function(prediction, score).to(device)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if epoch % 1 == 0:
                    print("epoch: {}, loss is: {}".format(epoch, loss.item()))
        torch.save(self.model.state_dict(), "score_roberta.pth")


linear = Linear_Model()
data = load_dataset('csv', data_files='./generate_data/sentenceSection_score.csv')
dataset = data['train']
data = dataset.train_test_split(test_size=0.2)
# print(data['train'])
linear.train(data)


'''
Add the corresponding section to each sentence in sentence_score.csv to obtain file sentenceSection_score.csv
'''

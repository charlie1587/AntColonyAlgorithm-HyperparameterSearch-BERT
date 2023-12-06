import json
import numpy as np
from torch.utils.data import Dataset
from transformers import BertTokenizer


# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, text, label):
        self.text_data = text
        self.label_data = label

    def __len__(self):
        return len(self.text_data)

    def get_batch_labels(self, idx):
        # Fetch a batch of labels
        return np.array(self.label_data[idx])

    def get_batch_texts(self, idx):
        # Fetch a batch of inputs
        return self.text_data[idx]

    def __getitem__(self, idx):
        batch_x = self.get_batch_texts(idx)
        batch_y = self.get_batch_labels(idx)
        return batch_x, batch_y


def read_data_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        f = f.readlines()
    # 用json格式把train拆分成text和label
    text = []
    label = []
    for i in f:
        text.append(json.loads(i)['text'])
        label.append(json.loads(i)['label'])
    return text, label


def get_data(conf, max_length=512):
    # 读取dev.txt, test.txt, train.txt
    dev_text, dev_label = read_data_from_file( conf['data_path'] + '/dev.txt')
    test_text, test_label = read_data_from_file( conf['data_path'] + '/test.txt')
    train_text, train_label = read_data_from_file( conf['data_path'] + '/train.txt')

    # 创建一个label字典
    label_dict = {}
    for i, j in enumerate(set(train_label)):
        label_dict[j] = i

    # 将label转换为label_dict中的数字
    train_label = [label_dict[i] for i in train_label]
    dev_label = [label_dict[i] for i in dev_label]
    test_label = [label_dict[i] for i in test_label]
    # 将label转换为tensor
    train_label = [np.array(i) for i in train_label]
    dev_label = [np.array(i) for i in dev_label]
    test_label = [np.array(i) for i in test_label]

    # 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(conf['bert_path'])

    # 将文本转换为token
    test_text = [tokenizer(i, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
                 for i in test_text]
    dev_text = [tokenizer(i, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
                for i in dev_text]
    train_text = [tokenizer(i, return_tensors='pt', padding='max_length', max_length=max_length, truncation=True)
                  for i in train_text]

    # 创建数据集
    train_data = CustomDataset(train_text, train_label)
    dev_data = CustomDataset(dev_text, dev_label)
    test_data = CustomDataset(test_text, test_label)

    # 打印训练集和测试集条数
    print('train_data:', len(train_data))
    print('dev_data:', len(dev_data))
    print('test_data:', len(test_data))

    # 返回train, test, label_dict
    return train_data, dev_data, test_data, label_dict

import torch
from transformers import BertModel, BertForSequenceClassification
from peft import get_peft_model, LoraConfig, TaskType
from torch import nn
import tqdm
import copy


class BertClassifier(nn.Module):
    def __init__(self, conf, layer_num=12):
        super(BertClassifier, self).__init__()
        # 从文件下加载bert分类器
        my_bert = BertModel.from_pretrained(conf['bert_path'])
        self.embedding = my_bert.embeddings
        self.encoder = my_bert.encoder
        self.pooler = my_bert.pooler
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 13)
        self.ReLU = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        # 检查attention_mask的形状
        if len(attention_mask.shape) == 2:
            # 如果attention_mask的形状是[batch_size, sequence_length]，
            # 则修改它的形状为:
            # [batch_size, num_attention_heads, sequence_length, sequence_length]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.expand(-1, 12, input_ids.size(1), -1)
        elif len(attention_mask.shape) == 3:
            # 如果attention_mask的形状是[batch_size, 1, sequence_length]，
            # 则修改它的形状为:
            # [batch_size, num_attention_heads, sequence_length, sequence_length]
            attention_mask = attention_mask.unsqueeze(1)
            attention_mask = attention_mask.expand(-1, 12, -1, -1)
        # BERT模型的forward传播
        embedded = self.embedding(input_ids=input_ids)
        encoded_layers = self.encoder(embedded, attention_mask=attention_mask)
        pooled_output = self.pooler(encoded_layers.last_hidden_state)

        # 线性层的前向传播
        x = self.linear1(pooled_output)
        x = self.linear2(x)
        x = self.linear3(x)
        x = self.ReLU(x)

        return x

class Classifier(object):
    def __init__(self, conf, layer_num=12):
        self.conf = conf  # 配置信息
        self.model = BertClassifier(conf, layer_num)  # 模型
        # 选择优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.conf['lr'], weight_decay=self.conf['weight_decay'])
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.conf['scheduler_step'],
                                                         gamma=self.conf['scheduler_gamma'])

        if torch.cuda.is_available() and self.conf['use_gpu']:
            print('use cuda for model '+str(self.conf['cuda_number']))
            torch.cuda.set_device(self.conf['cuda_number'])
            self.model = self.model.cuda()
        print("Bert Classifier init finished")

    def train(self, train_loader):
        train_loader = tqdm.tqdm(train_loader)
        for batch_id, batch in enumerate(train_loader):
            text_data, label_data = batch
            # 把text_data拆分成input_ids和attention_mask
            input_ids = text_data['input_ids']
            input_ids = input_ids.squeeze(1)
            attention_mask = text_data['attention_mask']
            if torch.cuda.is_available() and self.conf['use_gpu']:
                torch.cuda.set_device(self.conf['cuda_number'])
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                label_data = label_data.cuda()

            self.optimizer.zero_grad()  # 优化器置零
            output = self.model(input_ids, attention_mask)  # 获得预测结果
            loss = torch.nn.functional.cross_entropy(output, label_data)
            loss.backward()  # 进行反向传播
            self.optimizer.step()
        self.scheduler.step()

    def eval(self, data_loader):
        self.model.eval()
        losses = 0  # 记录损失
        correct = 0  # 记录正确数目
        dataset_size = 0  # 测试数据总数
        for batch_id, batch in enumerate(data_loader):  # 对测试数据进行编号和按batch提取数据
            text_data, label_data = batch  # 解包数据和标签
            # 把text_data拆分成input_ids和attention_mask
            input_ids = text_data['input_ids']
            input_ids = input_ids.squeeze(1)
            attention_mask = text_data['attention_mask']
            attention_mask = attention_mask.squeeze(1)

            # 把input_ids和attention_mask转换为tensor
            if torch.cuda.is_available() and self.conf['use_gpu']:
                torch.cuda.set_device(self.conf['cuda_number'])
                input_ids = input_ids.cuda()
                attention_mask = attention_mask.cuda()
                label_data = label_data.cuda()

            output = self.model(input_ids, attention_mask)  # 获得预测结果
            losses += torch.nn.functional.cross_entropy(output, label_data, reduction='sum').item()
            output = output.argmax(dim=1)  # 获得预测类别
            dataset_size += len(label_data)  # 记录测试数据总数
            # 遍历判断预测类别和真实类别是否相等
            matches = (output == label_data)
            correct += matches.sum()  # 记录正确数目

        acc = 100.0 * (float(correct) / float(dataset_size))  # 得到准确率的百分值
        # 打印准确率
        print('测试集准确率：{:.2f}'.format(acc))
        print('测试集损失：{:.2f}'.format(losses))
        return acc, losses

    def print_trainable_parameters(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.size())
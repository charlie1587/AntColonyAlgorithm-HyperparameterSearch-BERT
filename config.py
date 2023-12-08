import dataset
import classifier
from torch.utils.data import DataLoader

# 所有的参数
class conf:
    def __init__(self):
        # GPU参数
        self['use_gpu'] = True
        self['cuda_number'] = 0

        # 数据集参数
        self['data_path'] = "/home/tcq/PythonProject/116BERT/data"

        # BERT参数
        self['bert_path'] = "/home/tcq/PythonProject/116BERT/model/bert-base-uncased"
        self['unfreeze_layers'] = ['encoder.layer.11', 'encoder.layer.10', 'encoder.layer.9',
                                   'linear1', 'linear2', 'linear3']
        self['batch_size'] = 32
        self['epoch'] = 5
        self['lr'] = 5e-5
        self['eval_step'] = 1
        self['scheduler_step'] = 3
        self['scheduler_gamma'] = 1
        self['weight_decay'] = 1e-5

        # 蚁群算法参数
        self['num_ants'] = 2
        self['iterations'] = 2
        self['lower_bounds'] = [1e-5, 0.1]
        self['upper_bounds'] = [5e-5, 1]

        # 优化目标函数
        def objective_function(lr, scheduler_gamma):
            # 加载配置文件
            conf = self
            conf['lr'] = lr
            conf['scheduler_gamma'] = scheduler_gamma
            # 载入数据
            train_data, dev_data, test_data, label_dict = dataset.get_data(conf, 512)
            train_loader = DataLoader(train_data, batch_size=conf['batch_size'])
            test_loader = DataLoader(test_data, batch_size=conf['batch_size'])
            # 加载bert
            my_bert = classifier.Classifier(conf)
            # 测试bert
            for i in range(conf['epoch']):
                print("Epoch:", i + 1)
                my_bert.train(train_loader)
            acc, loss = my_bert.eval(test_loader)

            return loss

        self['objective_function'] = objective_function

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

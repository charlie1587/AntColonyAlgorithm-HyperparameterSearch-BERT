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
        self['batch_size'] = 32
        self['epoch'] = 15
        self['lr'] = 1e-5
        self['eval_step'] = 1
        self['scheduler_step'] = 2
        self['scheduler_gamma'] = 0.1
        self['weight_decay'] = 0.01

        # 蚁群算法参数
        self['num_ants'] = 10
        self['iterations'] = 1000
        self['lower_bounds'] = [-10, -10]
        self['upper_bounds'] = [10, 10]

        # 优化目标函数
        def objective_function(x, y):
            return (x - 3) ** 2 + (y - 5) ** 2  # 以(x=3, y=5)为最小值的二元函数

        self['objective_function'] = objective_function

    def __getitem__(self, item):
        return getattr(self, item)

    def __setitem__(self, key, value):
        setattr(self, key, value)

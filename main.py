from transformers import BertModel, BertForSequenceClassification
from torch.utils.data import DataLoader
import config
import ant
import dataset
import classifier

if __name__ == '__main__':
    # 加载配置文件
    conf = config.conf()
    # 载入数据
    train_data, dev_data, test_data, label_dict = dataset.get_data(conf, 512)
    train_loader = DataLoader(train_data, batch_size=conf['batch_size'])
    test_loader = DataLoader(test_data, batch_size=conf['batch_size'])
    # 加载bert
    my_bert = classifier.Classifier(conf)
    # 测试bert
    my_bert.train(train_loader)
    my_bert.eval(test_loader)
    # 测试蚁群算法
    ant_colony = ant.AntColony(num_ants=conf['num_ants'], iterations=conf['iterations'],
                               lower_bounds=conf['lower_bounds'], upper_bounds=conf['upper_bounds'],
                               objective_function=conf['objective_function'])
    best_solution, best_value = ant_colony.search()

    print("Best Solution:", best_solution)
    print("Best Value:", best_value)
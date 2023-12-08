import config
import ant


if __name__ == '__main__':
    # 加载配置文件
    conf = config.conf()
   # 蚁群算法
    ant_colony = ant.AntColony(num_ants=conf['num_ants'], iterations=conf['iterations'],
                               lower_bounds=conf['lower_bounds'], upper_bounds=conf['upper_bounds'],
                               objective_function=conf['objective_function'])
    best_solution, best_value = ant_colony.search()

    print("Best Solution:", best_solution)
    print("Best Value:", best_value)

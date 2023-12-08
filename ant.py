import random
import math


# 蚁群类
class AntColony:
    def __init__(self, num_ants, iterations, lower_bounds, upper_bounds, objective_function):
        self.num_ants = num_ants
        self.iterations = iterations
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.objective_function = objective_function
        self.best_solution = None
        self.best_value = float('inf')
        self.pheromones = [[1.0 for _ in range(len(lower_bounds))] for _ in range(num_ants)]
        print("Ant Colony init finished")
        print("Ant num:", num_ants)
        print("Iteration num:", iterations)

    # 更新信息素
    def update_pheromones(self, solutions, values):
        evaporation = 0.1
        for i in range(self.num_ants):
            for j in range(len(self.lower_bounds)):
                self.pheromones[i][j] *= (1 - evaporation)
            for idx, value in enumerate(solutions[i]):
                self.pheromones[i][idx] += 1.0 / (values[i] + 1e-10)

    # 生成解
    def generate_solution(self):
        solution = [random.uniform(self.lower_bounds[i], self.upper_bounds[i]) for i in range(len(self.lower_bounds))]
        return solution

    # 蚁群算法
    def search(self):
        for _ in range(self.iterations):
            solutions = [self.generate_solution() for _ in range(self.num_ants)]
            values = [self.objective_function(*sol) for sol in solutions]

            for i, sol in enumerate(solutions):
                if values[i] < self.best_value:
                    self.best_solution = sol
                    self.best_value = values[i]

            self.update_pheromones(solutions, values)

        return self.best_solution, self.best_value



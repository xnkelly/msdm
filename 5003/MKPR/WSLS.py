# last_restaurant_queues == 0

import numpy as np
import matplotlib.pyplot as plt

class KPRGameModifiedQueue:
    def __init__(self, N: int, M: int, q_star: int, seed: int = 42):
        self.N = N  # 顾客数量
        self.M = M  # 天数
        self.q_star = q_star  # 最大队列长度
        self.seed = seed  # 随机种子
        np.random.seed(self.seed)
        self.f_values = self.simulate_modified_kpr()

    def simulate_modified_kpr(self) -> np.ndarray:
        fractions = np.zeros(self.M)
        # 每个顾客在上一天的选择（初始随机选择餐厅）
        successful_customer = 0
        last_restaurant_queues = np.zeros(self.N, dtype=int)
        for day in range(self.M):
            # 当天餐厅队列
            choices_today = np.zeros(self.N, dtype=int)
    
            # 第一天天的初始化：每个顾客随机选择餐厅
            if day == 0:
                for customer in range(self.N):
                    available_restaurants = np.where(choices_today < self.q_star)[0]
                    choice = np.random.choice(available_restaurants)
                    choices_today[choice] += 1
            else:
                # 顾客按策略选择餐厅
                for customer in range(self.N):
                    # 如果前一天成功，就继续选择相同的餐厅（Win-Stay）
                    if customer < successful_customer and choices_today[customer] < self.q_star:
                        # 继续选择上一天的餐厅
                        choices_today[customer] += 1
                    else:
                        # Lose-Shift: 如果前一天失败，选择上一天队列为空的餐厅
                        vacant_restaurants = np.where((last_restaurant_queues == 0) & (choices_today < self.q_star))[0]
                        if len(vacant_restaurants) > 0:
                            choice = np.random.choice(vacant_restaurants)
                        else:
                            # 如果没有空餐厅，选择队列未满的餐厅
                            print("No vacant restaurants")
                            available_restaurants = np.where(choices_today < self.q_star)[0]
                            choice = np.random.choice(available_restaurants)
                        # 记录当天选择的餐厅
                        choices_today[choice] += 1
    
            # 计算占用的餐厅数量
            successful_customer = np.sum(choices_today > 0)  # 记录成功的顾客数
            fractions[day] = successful_customer / self.N
            last_restaurant_queues = np.copy(choices_today)
    
        return fractions
        
    def plot_daily_fractions(self):
        plt.plot(range(self.M), self.f_values, marker='o', linestyle='--', markersize=4, label=f'q*={self.q_star}')
        plt.xlabel('Day')
        plt.ylabel('Fraction of Occupied Restaurants')
        plt.title('Daily Occupancy Fraction Over Time')
        plt.ylim(0, 1)

def main():
    N = 512  # 顾客数量
    M = 40  # 模拟天数
    queue_lengths = [2, 3, 4, 5]  # 队列长度
    seed = 42

    plt.figure(figsize=(8, 6))

    for q_star in queue_lengths:
        game_modified_kpr = KPRGameModifiedQueue(N, M, q_star, seed)
        game_modified_kpr.plot_daily_fractions()

    plt.legend(['q*=' + str(q) for q in queue_lengths])
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()

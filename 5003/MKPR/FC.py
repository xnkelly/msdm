N = 512
M = 10**6
queue_lengths = [5]  
r_values = [0.1, 1.0, 9.0]  # 不同的 r 值
seed = 42

import numpy as np
import time

def simulate_FC_with_queue(N, T, q_star, r, seed=42):
    np.random.seed(seed)
    
    # 用于记录每天每个餐馆的队列长度 (T天，每天N个餐馆的队列)
    # q[t, k] 表示第t天餐馆k的队列长度
    q = np.zeros((T, N), dtype=int)
    
    # 第一天的初始化：每个顾客随机选择餐馆
    choices_today = np.zeros(N, dtype=int)  # 用于记录今天每个餐馆的顾客数
    cur_time = time.time()
    for customer in range(N):
        # 选择队列未满的餐馆
        available_restaurants = np.where(choices_today < q_star)[0]
        choice = np.random.choice(available_restaurants)
        choices_today[choice] += 1  # 该餐馆的队列长度+1

    # 将第一天的餐馆队列长度记录到q[0]
    q[0] = choices_today
    
    # 从第二天开始应用FC策略
    for day in range(1, T):
        # 上一天的队列长度
        yesterday_counts = q[day-1]
        
        # 根据公式：p_k(t+1) = (q_k(t) + r) / [N(1 + r)]
        probabilities = (yesterday_counts + r) / ((1 + r) * N)
        probabilities /= probabilities.sum()
        
        # 根据probabilities抽样N个顾客的餐馆选择
        choices = np.random.choice(N, size=N, p=probabilities)
        counts = np.bincount(choices, minlength=N)

        # 超额处理
        over = counts > q_star
        extras = counts - q_star
        extras[~over] = 0
        total_extras = extras.sum()

        if total_extras > 0:
            # 对每个超额顾客，尝试将其分配到队列未满的餐馆
            for _ in range(total_extras):
                # 选择队列未满的餐馆
                available_restaurants = np.where(counts < q_star)[0]
                
                if len(available_restaurants) == 0:
                    raise ValueError(f"Day {day}: No available restaurants with space to assign extra customers.")
                
                # 随机选择一个餐馆
                choice = np.random.choice(available_restaurants)
                counts[choice] += 1  # 增加该餐馆的队列长度

        # 更新当天的队列
        q[day] = counts
        if ((day+1)%100000==0):
                print(f"Day {day + 1}, time: {time.time()-cur_time}")
                cur_time=time.time()
    return q

import matplotlib.pyplot as plt

for q_star in queue_lengths:
        plt.figure(figsize=(10, 6))
    
        # 定义标记和颜色
        markers = ['o', 's', 'D', '^', 'x']
        colors = ['blue', 'green', 'orange', 'red', 'purple']
        mean_fractions = []
    
        for i, r in enumerate(r_values):
            q = simulate_FC_with_queue(N, M, q_star, r)
            fractions = (q > 0).sum(axis=1) / N
            unique_values, counts = np.unique(fractions, return_counts=True)
            probabilities = counts / M
    
            # 计算并记录均值
            mean_fraction = np.mean(fractions)
            mean_fractions.append(mean_fraction)
    
            plt.scatter(unique_values, probabilities, 
                        label=f'r = {r}', 
                        marker=markers[i], 
                        color=colors[i], 
                        alpha=0.6)
    
        plt.title(f'Utilization Fraction with Different r Values when $q^* = {q_star}$')
        plt.xlabel('Fraction of People, f')
        plt.ylabel('Probability, D(f)')
        plt.legend()
        plt.ylim(bottom=0)
        #plt.savefig(f"E:\\港科\\fall term\\5003\\final project\\fraction_distribution_fc(q={queue_length}).png")
        plt.savefig("FC-1.png")
        plt.show()
    
        # 输出均值
        for r, mean in zip(r_values, mean_fractions):
            print(f'For r = {r}, mean fraction = {mean:.4f}')

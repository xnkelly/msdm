import numpy as np
import matplotlib.pyplot as plt
import time

class KPRGameModifiedQueue:
    def __init__(self, N: int, M: int, q_star: int, seed: int = 42):
        self.N = N  # 顾客数量
        self.M = M  # 天数
        self.q_star = q_star  # 最大队列长度
        self.seed = seed  # 随机种子
        np.random.seed(self.seed)
        self.f_values = self.simulate_modified_kpr()

    def simulate_modified_kpr(self) -> np.ndarray:
        start_time=time.time()
        cur_time=time.time()
        fractions = np.zeros(self.M)
        # 每个顾客在上一天的选择（初始随机选择餐厅）
        successful_customer = 0
        for day in range(self.M):
            # 当天餐厅队列初始化
            choices_today = np.zeros(self.N-successful_customer, dtype=int)
            if day == 0:
                # 第一天随机分配
                for customer in range(self.N):
                    available_restaurants = np.where(choices_today < self.q_star)[0]
                    choice = np.random.choice(available_restaurants)
                    choices_today[choice] += 1
            else:
                for customer in range(self.N-successful_customer):
                    # Win-Stay：上一日成功的顾客继续选择同一家餐馆
                    # Lose-Shift：未成功的顾客选择其余餐馆
                    available_restaurants = np.where(choices_today < self.q_star)[0]
                    choice = np.random.choice(available_restaurants)
                    choices_today[choice] += 1
        
            # 计算当天成功的餐馆（或成功顾客）
            successful_customer_today = np.sum(choices_today > 0) + 1   # 前一天成功顾客竞争同一家餐馆
            fractions[day] = successful_customer + successful_customer_today
            successful_customer = successful_customer_today
            if ((day + 1) % 50000 == 0):
                print(f"Day {day + 1}, Queue Length: {fractions[day]}, time: {time.time() - cur_time}")
                cur_time = time.time()

        return fractions                
            
def main():
    N = 1000  # 顾客数量
    M = 10**6  # 模拟天数
    queue_lengths = [2,3,5,1000]  # 队列长度
    seed = 42

    plt.figure(figsize=(10, 6))
    
    # Marker styles
    markers = ['+', 'x', 's', 'o']  # Marker types
    color = 'black'  # Uniform color
    for i, q_star in enumerate(queue_lengths):
        game_modified_kpr = KPRGameModifiedQueue(N, M, q_star, seed)
        fractions = game_modified_kpr.f_values
        f_mean = np.mean(fractions)
        # f_std = np.std(fractions)
        print(f"Average fraction of restaurants utilized (f̄) with OPR learning q*={q_star}: {f_mean/game_modified_kpr.N:.4f}")
        #print("Standard Deviation of f =", f_std)
    
        # Plot distribution
        # 计算 Δf = f - f̄
        delta_f = fractions - f_mean
        
        # 裸分布 D(f)
        bins = len(np.unique(delta_f))
        hist, bin_edges = np.histogram(delta_f, bins=bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        # plt.plot(bin_centers, hist, 's', markeredgecolor='k', markerfacecolor='none',  markersize=5, label='LL1: D(f)')
        
        # 归一化分布 D0(f)
        D0_f = hist / np.max(hist)  # 归一化分布
        #plt.plot(bin_centers, D0_f, 's', markeredgecolor='k', markerfacecolor='none',  markersize=5, label=f'q*={self.q_star}')
        
        # Choose the marker type based on the index i
        marker = markers[i]  # Loop through the markers if there are more queue_lengths than markers
    
        # Plot using scatter with different markers
        if marker == 's' or marker == 'o':  # If the marker is square or circle
            plt.scatter(bin_centers, D0_f, label=f'q* = {q_star}', 
                        marker=marker, edgecolor=color, facecolors='none')
        else:  # For the other marker types
            plt.scatter(bin_centers, D0_f, label=f'q* = {q_star}', 
                        marker=marker, color=color)

    #plt.title('Normalized Distribution $D_0(f)$ for Limited Learning (LL1) Strategy')
    plt.xlabel(r'$\Delta f$')
    plt.ylabel(r'$D_0(f)$')
    plt.xlim(-50, 90)
    #plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

import numpy as np
import matplotlib.pyplot as plt

class KPRGameModifiedQueue:
    def __init__(self, N: int, M: int, q_star: int, seed: int = 42):
        self.N = N
        self.M = M
        self.q_star = q_star
        self.seed = seed
        np.random.seed(self.seed)
        self.f_values = self.simulate_modified_kpr()
        self.f_bar = np.mean(self.f_values)

    def simulate_modified_kpr(self) -> np.ndarray:
        fractions = np.zeros(self.M)
        batch_size = 10000
        num_batches = self.M // batch_size
        remainder = self.M % batch_size

        for batch in range(num_batches):
            choices = np.random.randint(0, self.N, size=(batch_size, self.N))
            counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.N), axis=1, arr=choices)
            counts_after = counts.copy()

            over = counts > self.q_star
            extras = counts - self.q_star
            extras[~over] = 0

            total_extras = extras.sum(axis=1)
            available = (counts < self.q_star)
            available_slots = self.q_star - counts
            available_slots[~available] = 0

            available_restaurants = [np.repeat(np.where(row)[0], row[np.where(row)[0]]) for row in available_slots]
            available_restaurants_flat = [day_slots for day_slots in available_restaurants]

            for day in range(batch_size):
                if total_extras[day] == 0:
                    continue
                avail_rest = available_restaurants_flat[day]
                if len(avail_rest) < extras[day].sum():
                    raise ValueError(f"Day {batch * batch_size + day + 1}: Not enough available slots to assign extras.")
                new_assignments = np.random.choice(avail_rest, size=int(total_extras[day]), replace=False)
                counts_after[day, new_assignments] += 1

            occupied = np.count_nonzero(counts_after >= 1, axis=1)
            f_t = occupied / self.N
            fractions[batch * batch_size : (batch + 1) * batch_size] = f_t

        if remainder > 0:
            choices = np.random.randint(0, self.N, size=(remainder, self.N))
            counts = np.apply_along_axis(lambda x: np.bincount(x, minlength=self.N), axis=1, arr=choices)
            counts_after = counts.copy()

            over = counts > self.q_star
            extras = counts - self.q_star
            extras[~over] = 0

            total_extras = extras.sum(axis=1)
            available = (counts < self.q_star)
            available_slots = self.q_star - counts
            available_slots[~available] = 0

            available_restaurants = [np.repeat(np.where(row)[0], row[np.where(row)[0]]) for row in available_slots]
            available_restaurants_flat = [day_slots for day_slots in available_restaurants]

            for day in range(remainder):
                if total_extras[day] == 0:
                    continue
                avail_rest = available_restaurants_flat[day]
                if len(avail_rest) < extras[day].sum():
                    raise ValueError(f"Day {num_batches * batch_size + day + 1}: Not enough available slots to assign extras.")
                new_assignments = np.random.choice(avail_rest, size=int(total_extras[day]), replace=False)
                counts_after[day, new_assignments] += 1

            occupied = np.count_nonzero(counts_after >= 1, axis=1)
            f_t = occupied / self.N
            fractions[num_batches * batch_size :] = f_t

        return fractions

    def analyze_and_plot(self, marker, color, label):
        unique_values, counts = np.unique(self.f_values, return_counts=True)
        probabilities = counts / self.M
        plt.scatter(unique_values, probabilities, marker=marker, edgecolor=color, facecolors='none', label=label)

def main():
    N = 1024
    M = 10**6
    queue_lengths = [2, 3, 4, 5]
    seed = 5003

    plt.figure(figsize=(10, 6))

    markers = ['+', 'x', '*', 's']
    color = 'black'

    for i, q_star in enumerate(queue_lengths):
        game_modified_kpr = KPRGameModifiedQueue(N, M, q_star)
        unique_values, counts = np.unique(game_modified_kpr.f_values, return_counts=True)
        probabilities = counts / M

        # 计算均值
        mean_f = np.mean(game_modified_kpr.f_values)
        print(f'Mean f for q* = {q_star}: {mean_f:.4f}')

        if markers[i] == 's':
            plt.scatter(unique_values, probabilities, label=f'q* = {q_star}', 
                        marker=markers[i], edgecolor=color, facecolors='none')
        else:
            plt.scatter(unique_values, probabilities, label=f'q* = {q_star}', 
                        marker=markers[i], color=color)

    plt.title('Distribution of the Fraction of People Getting Lunch')
    plt.xlabel('Fraction of People, f')
    plt.ylabel('Probability, D(f)')
    plt.ylim(bottom=0)  
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()

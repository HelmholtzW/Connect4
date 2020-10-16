from matplotlib import pyplot as plt
import numpy as np
import time
import json
from statistics import stdev
import os


def main():
    file_path = "/Users/benbeinke/programming/Connect4/stats/to_plot"
    # std(file_path)

    # plot_folder(file_path)

    # scores, lr, bs, memory_size, update_rate, name = average(file_path)
    # plot(scores, lr, bs, memory_size, update_rate, name)
    multi_plot("/Users/benbeinke/programming/Connect4/stats/averages")


def std(file_path):
    final_scores = []
    for file in os.listdir(file_path):
        if file.endswith(".json"):
            with open(os.path.join(file_path, file), 'r') as f:
                file = json.load(f)
                score = file[0][500]
                final_scores.append(score)
    print(stdev(final_scores))


def plot_folder(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), 'r') as f:
                name = file[:-5]
                file = json.load(f)
                scores = file[0]
                lr = file[1]
                bs = file[2]
                memory_size = file[3]
                update_rate = file[4]
                plot(scores, lr, bs, memory_size, update_rate, name)


def multi_plot(folder_path):
    scores = []
    for file in os.listdir(folder_path):
        if file.endswith(".json"):
            with open(os.path.join(folder_path, file), 'r') as f:
                name = file[:-5]
                file = json.load(f)
                scores.append(file)
    m_plot(scores)


def m_plot(score_list):
        # markers = 'x', '^', 'o'
    plt.figure()
    plt.plot(score_list[0], 'g-', label="QVA-Learning")
    plt.plot(score_list[1], 'b-', label="QV-Learning")
    plt.plot(score_list[2], 'r-', label="Q-Learning")
    plt.legend()
    plt.axis([0, len(score_list[0]) - 1, 0, 100])
    plt.xlabel("Number of Games * 100")
    plt.ylabel("Scores")
    plt.title(f"Learning Curves")
    plt.savefig("/Users/benbeinke/programming/Connect4/" +
                "Q_vs_QV_vs_QVA" + '.png', dpi=300)


def plot(score_list, lr, bs, memory_size, update_rate, name):
        # markers = 'x', '^', 'o'
    plt.figure()
    plt.plot(score_list, 'b-')
    plt.axis([0, len(score_list) - 1, 0, 100])
    plt.xlabel("Number of Games * 100")
    plt.ylabel("Scores")
    plt.title(f"Q-Learning vs Benchmark lr:{lr}, bs={bs} ms={memory_size}, ur={update_rate}")
    plt.savefig("/Users/benbeinke/programming/Connect4/" +
                name + '.png', dpi=300)


def average(file_path):
    averages = np.zeros(501)
    i = 0
    for file in os.listdir(file_path):
        i += 1
        if file.endswith(".json"):
            with open(os.path.join(file_path, file), 'r') as f:
                name = file[:-5]
                file = json.load(f)
                scores = file[0]
                lr = file[1]
                bs = file[2]
                memory_size = file[3]
                update_rate = file[4]
                averages += scores

    averages = averages / i
    with open(os.path.join(file_path, f"Average_{time.time()}.json"), "w") as f:
        json.dump(averages.tolist(), f)

    return averages, lr, bs, memory_size, update_rate, f"Average_{time.time()}.json"


if __name__ == "__main__":
    main()

import matplotlib.pyplot as plt
import random


def gererate_random_walk(length=100, mu=0, sig=1):
    ts = []
    for i in range(length):
        e = random.gauss(mu, sig)
        if i == 0:
            ts.append(e)
        else:
            ts.append(ts[i - 1] + e)
    return ts


if __name__ == '__main__':
    random.seed(5)
    random_walk = gererate_random_walk(length=250)
    plt.plot(random_walk)
    plt.show()

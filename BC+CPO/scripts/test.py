import numpy as np

a=np.array([[1,2,3],[4,5,6],[7,8,9]])
b=np.array([[5,6,7],[1,2,3]])


def gaussian_likelihood(x, mu, log_std):
    # std = np.exp(log_std)
    # p = 1 / (np.sqrt(2 * np.pi) * std) * np.exp(-(x - mu) ** 2 / (2 * std ** 2))
    pre_sum = -0.5 * (((x - mu) / (np.exp(log_std) + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return np.mean(pre_sum)

x=np.array(-0.56817627)
mu=np.array(-1.836)
log_std=np.array(-0.5)
print(gaussian_likelihood(x, mu, log_std))

x=np.array(1)
mu=np.array(1.070933)
log_std=np.array(-0.5)
print(gaussian_likelihood(x, mu, log_std))
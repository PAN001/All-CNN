import numpy as np

def shuffle(imgs, labels, seed = 1):
    # np.random.seed(seed)
    permutation_index = np.random.permutation(range(0, len(imgs)))
    imgs = imgs[permutation_index]
    labels = labels[permutation_index]

    return imgs, labels
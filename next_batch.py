def next_batch(imgs, labels, batch, batch_size):
    batch_xs = imgs[batch * batch_size : (batch + 1) * batch_size]
    batch_ys = labels[batch * batch_size: (batch + 1) * batch_size]
    return batch_xs, batch_ys
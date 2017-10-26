import pickle
import numpy as np

# l = [0.111223123231,-.123123123123]
# with open("test.txt", "wb") as fp:
#     pickle.dump(l, fp)

with open("glorot_uniform/all_cnn_val_accs_epoch_glorot_uniform.acc", "rb") as fp:
    b = pickle.load(fp)
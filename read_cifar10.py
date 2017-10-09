import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pickle
# from PIL import Image
import matplotlib.pyplot as plt

def reshape_cifar10(imgs_raw):
    # labels_ndarray = np.array(labels_raw)
    imgs_reshaped = np.array(imgs_raw)
    imgs_reshaped = imgs_reshaped.reshape(len(imgs_reshaped), 32, 32, 3)
    # imgs_display = np.transpose(np.reshape(imgs_reshaped, (len(imgs_reshaped), 3, 32, 32)), (0, 2, 3, 1))

    return imgs_reshaped


# fp = "./cifar-10-batches-py/data_batch_1"
# f = open(fp, 'rb')

# tupled_data= pickle.load(f)
#
# f.close()
# img = tupled_data[b'data']
# label = tupled_data[b'labels']
# single_img = np.array(img[5])
# single_img_reshaped = single_img.reshape(32,32,3)
# single_img_reshaped = np.transpose(np.reshape(single_img,(3, 32,32)), (1,2,0))
#
# plt.imshow(single_img_reshaped)
# plt.show()

path = "./cifar-10-batches-py/"
ds = {}
for i in range(5):
    print i
    with open(path+"/data_batch_"+str(i+1), "rb") as db:
        db_set = pickle.load(db)

        # reshape_cifar10
        imgs_raw = db_set[b"data"] # list of images with shape of (3072, 1)
        labels_raw = db_set[b"labels"]  # list of labels
        # labels_ndarray = np.array(labels_raw)
        # imgs_reshaped = np.array(imgs_raw)
        # imgs_reshaped = imgs_reshaped.reshape(len(imgs_reshaped), 32, 32, 3)
        # imgs_display = np.transpose(np.reshape(imgs_reshaped, (len(imgs_reshaped), 3, 32, 32)), (0, 2, 3, 1))
        ds["data_"+str(i+1)], ds["labels_"+str(i+1)] = reshape_cifar10(db_set[b"data"]), db_set[b"labels"]
        ds["data_display_" + str(i + 1)] = np.transpose(np.reshape(ds["data_"+str(i+1)], (len(ds["data_"+str(i+1)]), 3, 32, 32)), (0, 2, 3, 1))

with open(path+"/test_batch", "rb") as tb:
    test_set = pickle.load(tb)
    ds["test_data"], ds["test_labels"] = reshape_cifar10(test_set[b"data"]), test_set[b"labels"]
    ds["test_data_display"] = np.transpose(np.reshape(ds["test_data"], (len(ds["test_data"]), 3, 32, 32)), (0, 2, 3, 1))

# concatenate batches to form training set
training_data = (ds["data_1"], ds["data_2"], ds["data_3"], ds["data_4"], ds["data_5"])
training_labels = (ds["labels_1"], ds["labels_2"], ds["labels_3"], ds["labels_4"], ds["labels_5"])
ds["training_data"] = np.concatenate(training_data, axis=0)
ds["training_labels"] = np.concatenate(training_labels, axis=0)
    # return ds
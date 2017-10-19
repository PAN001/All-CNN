import pickle
import numpy as np

def reshape_cifar10(imgs_raw):
    imgs_reshaped = np.array(imgs_raw)
    imgs_reshaped = imgs_reshaped.reshape(len(imgs_reshaped), 32, 32, 3)

    return imgs_reshaped

def load_cifar10(path):
    ds = {}
    for i in range(5):
        with open(path+"/data_batch_"+str(i+1), "rb") as db:
             db_set = pickle.load(db)
             ds["data_"+str(i+1)], ds["labels_"+str(i+1)] = reshape_cifar10(db_set[b"data"]), np.array(db_set[b"labels"])

    with open(path+"/test_batch", "rb") as tb:
        test_set = pickle.load(tb)
        ds["test_data"], ds["test_labels"] = reshape_cifar10(test_set[b"data"]), np.array(test_set[b"labels"])
        ds["data_display_" + str(i + 1)] = np.transpose(np.reshape(ds["data_" + str(i + 1)], (len(ds["data_" + str(i + 1)]), 3, 32, 32)), (0, 2, 3, 1))
        ds["test_data_display"] = np.transpose(np.reshape(ds["test_data"], (len(ds["test_data"]), 3, 32, 32)), (0, 2, 3, 1))

    # concatenate batches to form training set
    training_data = (ds["data_1"], ds["data_2"], ds["data_3"], ds["data_4"], ds["data_5"])
    training_labels = (ds["labels_1"], ds["labels_2"], ds["labels_3"], ds["labels_4"], ds["labels_5"])
    ds["training_data"] = np.concatenate(training_data, axis=0)
    ds["training_labels"] = np.concatenate(training_labels, axis=0)
    return ds
import matplotlib.pyplot as plt
import pylab as pl
import pickle

# id = "LSUV"
# accs_epoch_path = id + "/" + "all_cnn_accs_epoch_" + id + ".acc"
# losses_epoch_path = id + "/" + "all_cnn_losses_epoch_" + id + ".loss"
# val_accs_epoch_path = id + "/" + "all_cnn_val_accs_epoch_" + id + ".acc"
# val_losses_epoch_path = id + "/" + "all_cnn_val_losses_epoch_" + id + ".acc"
# accs_batch_path = "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path = "all_cnn_losses_batch_" + id + ".loss"
#
# with open(accs_epoch_path, "rb") as fp:
#     accs_epoch = pickle.load(fp)
#
# with open(losses_epoch_path, "rb") as fp:
#     losses_epoch = pickle.load(fp)
#
# with open(val_accs_epoch_path, "rb") as fp:
#     val_accs_epoch = pickle.load(fp)
#
# with open(val_losses_epoch_path, "rb") as fp:
#     val_losses_epoch = pickle.load(fp)
#
# with open(accs_batch_path, "rb") as fp:
#     accs_batch = pickle.load(fp)
#
# with open(losses_batch_path, "rb") as fp:
#     losses_batch = pickle.load(fp)

# plt.plot(accs_epoch)
# plt.plot(val_accs_epoch)
# plt.title('model accuracy per epoch')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # plt.savefig(acc_figure_path)
#
# plt.plot(losses_epoch)
# plt.plot(val_losses_epoch)
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # plt.savefig(loss_figure_path)

# fig = plt.figure(1, figsize=(20, 5))
# plt.plot(range(0, len(accs_batch))[0:-1:5], accs_batch[0:-1:5])
# plt.title(id + ': model accuracy per batch')
# plt.ylabel('acc')
# plt.xlabel('batch')
# # plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # plt.savefig(loss_figure_path)
#
# fig = plt.figure(2, figsize=(20, 5))
# plt.plot(range(0, len(losses_batch))[0:-1:5], losses_batch[0:-1:5])
# plt.title(id + ': model loss per batch')
# plt.ylabel('loss')
# plt.xlabel('batch')
# # plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# # plt.savefig(loss_figure_path)


id = "LSUV"
accs_batch_path_LSUV = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
losses_batch_path_LSUV = id + "/" + "all_cnn_losses_batch_" + id + ".loss"

id = "glorot_uniform"
accs_batch_path_glorot = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
losses_batch_path_glorot = id + "/" + "all_cnn_losses_batch_" + id + ".loss"

id = "he_uniform"
accs_batch_path_he = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
losses_batch_path_he = id + "/" + "all_cnn_losses_batch_" + id + ".loss"


with open(accs_batch_path_LSUV, "rb") as fp:
    accs_batch_LSUV = pickle.load(fp)

with open(losses_batch_path_LSUV, "rb") as fp:
    losses_batch_LSUV = pickle.load(fp)

with open(accs_batch_path_glorot, "rb") as fp:
    accs_batch_glorot = pickle.load(fp)

with open(losses_batch_path_glorot, "rb") as fp:
    losses_batch_glorot = pickle.load(fp)

with open(accs_batch_path_he, "rb") as fp:
    accs_batch_he = pickle.load(fp)

with open(losses_batch_path_he, "rb") as fp:
    losses_batch_he = pickle.load(fp)

# acc
fig = plt.figure(1, figsize=(40, 10))
plt.plot(range(0, len(accs_batch_LSUV))[0:-1:5], accs_batch_LSUV[0:-1:5])
plt.plot(range(0, len(accs_batch_glorot))[0:-1:5], accs_batch_glorot[0:-1:5])
plt.plot(range(0, len(accs_batch_he))[0:-1:5], accs_batch_he[0:-1:5])
plt.title('Exp1: model accuracy per batch')
plt.ylabel('acc')
plt.xlabel('batch')
plt.legend(['LSUV', 'Glorot uniform', 'He uniform'], loc='upper left')
# plt.show()
plt.savefig("exp1_acc.png")


# loss
fig = plt.figure(2, figsize=(40, 10))
plt.plot(range(0, len(losses_batch_LSUV))[0:-1:5], losses_batch_LSUV[0:-1:5])
plt.plot(range(0, len(losses_batch_glorot))[0:-1:5], losses_batch_glorot[0:-1:5])
plt.plot(range(0, len(losses_batch_he))[0:-1:5], losses_batch_he[0:-1:5])
plt.title('Exp1: model loss per batch')
plt.ylabel('loss')
plt.xlabel('batch')
plt.legend(['LSUV', 'Glorot uniform', 'He uniform'], loc='upper left')
# plt.show()
plt.savefig("exp1_loss.png")
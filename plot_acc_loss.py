import matplotlib.pyplot as plt
import pylab as pl
import pickle

id = "glorot_uniform"
accs_epoch_path = id + "/" + "all_cnn_accs_epoch_" + id + ".acc"
losses_epoch_path = id + "/" + "all_cnn_losses_epoch_" + id + ".loss"
val_accs_epoch_path = id + "/" + "all_cnn_val_accs_epoch_" + id + ".acc"
val_losses_epoch_path = id + "/" + "all_cnn_val_losses_epoch_" + id + ".acc"
accs_batch_path = "all_cnn_accs_batch_" + id + ".acc"
losses_batch_path = "all_cnn_losses_batch_" + id + ".loss"

with open(accs_epoch_path, "rb") as fp:
    accs_epoch = pickle.load(fp)

with open(losses_epoch_path, "rb") as fp:
    losses_epoch = pickle.load(fp)

with open(val_accs_epoch_path, "rb") as fp:
    val_accs_epoch = pickle.load(fp)

with open(val_losses_epoch_path, "rb") as fp:
    val_losses_epoch = pickle.load(fp)

with open(accs_batch_path, "rb") as fp:
    accs_batch = pickle.load(fp)

with open(losses_batch_path, "rb") as fp:
    losses_batch = pickle.load(fp)

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

plt.plot(accs_batch)
plt.title('model accuracy per batch')
plt.ylabel('acc')
plt.xlabel('batch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
# plt.savefig(loss_figure_path)

plt.plot(losses_batch)
plt.title('model loss per batch')
plt.ylabel('loss')
plt.xlabel('batch')
# plt.legend(['train', 'test'], loc='upper left')
plt.show()
# plt.savefig(loss_figure_path)
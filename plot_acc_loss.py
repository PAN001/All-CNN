import matplotlib.pyplot as plt
import pylab as pl
import pickle
import csv

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

# # exp1
# id = "LSUV"
# accs_batch_path_LSUV = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
# id = "glorot_uniform"
# accs_batch_path_glorot = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_glorot = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
# id = "he_uniform"
# accs_batch_path_he = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_he = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
#
# with open(accs_batch_path_LSUV, "rb") as fp:
#     accs_batch_LSUV = pickle.load(fp)
#
# with open(losses_batch_path_LSUV, "rb") as fp:
#     losses_batch_LSUV = pickle.load(fp)
#
# with open(accs_batch_path_glorot, "rb") as fp:
#     accs_batch_glorot = pickle.load(fp)
#
# with open(losses_batch_path_glorot, "rb") as fp:
#     losses_batch_glorot = pickle.load(fp)
#
# with open(accs_batch_path_he, "rb") as fp:
#     accs_batch_he = pickle.load(fp)
#
# with open(losses_batch_path_he, "rb") as fp:
#     losses_batch_he = pickle.load(fp)
#
# # acc
# fig = plt.figure(1, figsize=(40, 10))
# plt.plot(range(0, len(accs_batch_LSUV))[0:-1:5], accs_batch_LSUV[0:-1:5])
# plt.plot(range(0, len(accs_batch_glorot))[0:-1:5], accs_batch_glorot[0:-1:5])
# plt.plot(range(0, len(accs_batch_he))[0:-1:5], accs_batch_he[0:-1:5])
# plt.title('Exp1: model accuracy per batch')
# plt.ylabel('acc')
# plt.xlabel('batch')
# plt.legend(['LSUV', 'Glorot uniform', 'He uniform'], loc='upper left')
# # plt.show()
# plt.savefig("exp1_acc.png")
#
#
# # loss
# fig = plt.figure(2, figsize=(40, 10))
# plt.plot(range(0, len(losses_batch_LSUV))[0:-1:5], losses_batch_LSUV[0:-1:5])
# plt.plot(range(0, len(losses_batch_glorot))[0:-1:5], losses_batch_glorot[0:-1:5])
# plt.plot(range(0, len(losses_batch_he))[0:-1:5], losses_batch_he[0:-1:5])
# plt.title('Exp1: model loss per batch')
# plt.ylabel('loss')
# plt.xlabel('batch')
# plt.legend(['LSUV', 'Glorot uniform', 'He uniform'], loc='upper left')
# # plt.show()
# plt.savefig("exp1_loss.png")

# # exp2
# id = "LSUV_nopp"
# accs_batch_path_LSUV_nopp = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_nopp = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
# id = "LSUV_norm_shift_flip"
# accs_batch_path_LSUV_nsf = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_nsf = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
# id = "LSUV_norm_shift_flip_zca"
# accs_batch_path_LSUV_norm_shift_flip_zca = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_norm_shift_flip_zca = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
#
# with open(accs_batch_path_LSUV_nopp, "rb") as fp:
#     accs_batch_LSUV_nopp = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_nopp, "rb") as fp:
#     losses_batch_LSUV_nopp = pickle.load(fp)
#
# with open(accs_batch_path_LSUV_nsf, "rb") as fp:
#     accs_batch_LSUV_nsf = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_nsf, "rb") as fp:
#     losses_batch_LSUV_nsf = pickle.load(fp)
#
# with open(accs_batch_path_LSUV_norm_shift_flip_zca, "rb") as fp:
#     accs_batch_LSUV_norm_shift_flip_zca = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_norm_shift_flip_zca, "rb") as fp:
#     losses_batch_LSUV_norm_shift_flip_zca = pickle.load(fp)
#
# # acc
# fig = plt.figure(1, figsize=(40, 10))
# plt.plot(range(0, len(accs_batch_LSUV_nopp))[0:-1:5], accs_batch_LSUV_nopp[0:-1:5])
# plt.plot(range(0, len(accs_batch_LSUV_nsf))[0:-1:5], accs_batch_LSUV_nsf[0:-1:5])
# plt.plot(range(0, len(accs_batch_LSUV_norm_shift_flip_zca))[0:-1:5], accs_batch_LSUV_norm_shift_flip_zca[0:-1:5])
# # plt.xticks(range(0, len(accs_batch_LSUV_nopp))[0:-1:5], range(0, len(accs_batch_LSUV_nopp))[0:-1:500])
# plt.title('Exp2: model accuracy per batch')
# plt.ylabel('acc')
# plt.xlabel('batch')
# plt.legend(['no preprocessing', 'shift, flip and normalization', 'shift, flip, normalization and zca whitening'], loc='upper left')
# # plt.show()
# plt.savefig("exp2_acc.png")
#
#
# # loss
# fig = plt.figure(2, figsize=(40, 10))
# plt.plot(range(0, len(losses_batch_LSUV_nopp))[0:-1:5], losses_batch_LSUV_nopp[0:-1:5])
# plt.plot(range(0, len(losses_batch_LSUV_nsf))[0:-1:5], losses_batch_LSUV_nsf[0:-1:5])
# plt.plot(range(0, len(losses_batch_LSUV_norm_shift_flip_zca))[0:-1:5], losses_batch_LSUV_norm_shift_flip_zca[0:-1:5])
# # plt.xticks(range(0, len(losses_batch_LSUV_nopp))[0:-1:5], range(0, len(losses_batch_LSUV_nopp))[0:-1:5])
# plt.title('Exp2: model loss per batch')
# plt.ylabel('loss')
# plt.xlabel('batch')
# plt.legend(['no preprocessing', 'shift, flip and normalization', 'shift, flip, normalization and zca whitening'], loc='upper left')
# # plt.show()
# plt.savefig("exp2_loss.png")


# # exp3
# id = "LSUV_dropout"
# accs_batch_path_LSUV_dropout = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_dropout = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
# id = "LSUV_no_dropout"
# accs_batch_path_LSUV_no_dropout = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_no_dropout = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
# id = "LSUV_bn"
# accs_batch_path_LSUV_bn = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_bn = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
#
# with open(accs_batch_path_LSUV_dropout, "rb") as fp:
#     accs_batch_LSUV_dropout = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_dropout, "rb") as fp:
#     losses_batch_LSUV_dropout = pickle.load(fp)
#
# with open(accs_batch_path_LSUV_no_dropout, "rb") as fp:
#     accs_batch_LSUV_no_dropout = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_no_dropout, "rb") as fp:
#     losses_batch_LSUV_no_dropout = pickle.load(fp)
#
# with open(accs_batch_path_LSUV_bn, "rb") as fp:
#     accs_batch_LSUV_bn = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_bn, "rb") as fp:
#     losses_batch_LSUV_bn = pickle.load(fp)
#
# # acc
# fig = plt.figure(1, figsize=(40, 10))
# plt.plot(range(0, len(accs_batch_LSUV_no_dropout))[0:-1:5], accs_batch_LSUV_no_dropout[0:-1:5])
# plt.plot(range(0, len(accs_batch_LSUV_dropout))[0:-1:5], accs_batch_LSUV_dropout[0:-1:5])
# plt.plot(range(0, len(accs_batch_LSUV_bn))[0:-1:5], accs_batch_LSUV_bn[0:-1:5])
# # plt.xticks(range(0, len(accs_batch_LSUV_nopp))[0:-1:5], range(0, len(accs_batch_LSUV_nopp))[0:-1:500])
# plt.title('Exp3: model accuracy per batch')
# plt.ylabel('acc')
# plt.xlabel('batch')
# plt.legend(['baseline', 'dropout', 'bn'], loc='upper left')
# # plt.show()
# plt.savefig("exp3_acc.png")
#
#
# # loss
# fig = plt.figure(2, figsize=(40, 10))
# plt.plot(range(0, len(losses_batch_LSUV_no_dropout))[0:-1:5], losses_batch_LSUV_no_dropout[0:-1:5])
# plt.plot(range(0, len(losses_batch_LSUV_dropout))[0:-1:5], losses_batch_LSUV_dropout[0:-1:5])
# plt.plot(range(0, len(losses_batch_LSUV_bn))[0:-1:5], losses_batch_LSUV_bn[0:-1:5])
# # plt.xticks(range(0, len(losses_batch_LSUV_nopp))[0:-1:5], range(0, len(losses_batch_LSUV_nopp))[0:-1:5])
# plt.title('Exp3: model loss per batch')
# plt.ylabel('loss')
# plt.xlabel('batch')
# plt.legend(['baseline', 'dropout', 'bn'], loc='upper left')
# # plt.show()
# plt.savefig("exp3_loss.png")

# # exp4
# id = "LSUV_no_dropout"
# accs_batch_path_LSUV_no_dropout = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_no_dropout = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
#
# id = "LSUV_rmsp"
# accs_batch_path_LSUV_rmsp = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_rmsp = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
# id = "LSUV_Adam"
# accs_batch_path_LSUV_adam = id + "/" + "all_cnn_accs_batch_" + id + ".acc"
# losses_batch_path_LSUV_adam = id + "/" + "all_cnn_losses_batch_" + id + ".loss"
#
# with open(accs_batch_path_LSUV_no_dropout, "rb") as fp:
#     accs_batch_LSUV_no_dropout = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_no_dropout, "rb") as fp:
#     losses_batch_LSUV_no_dropout = pickle.load(fp)
#
# with open(accs_batch_path_LSUV_rmsp, "rb") as fp:
#     accs_batch_LSUV_rmsp = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_rmsp, "rb") as fp:
#     losses_batch_LSUV_rmsp = pickle.load(fp)
#
# with open(accs_batch_path_LSUV_adam, "rb") as fp:
#     accs_batch_LSUV_adam = pickle.load(fp)
#
# with open(losses_batch_path_LSUV_adam, "rb") as fp:
#     losses_batch_LSUV_adam = pickle.load(fp)
#
# # acc
# fig = plt.figure(1, figsize=(40, 10))
# plt.plot(range(0, len(accs_batch_LSUV_no_dropout))[0:-1:5], accs_batch_LSUV_no_dropout[0:-1:5])
# plt.plot(range(0, len(accs_batch_LSUV_rmsp))[0:-1:5], accs_batch_LSUV_rmsp[0:-1:5])
# plt.plot(range(0, len(accs_batch_LSUV_adam))[0:-1:5], accs_batch_LSUV_adam[0:-1:5])
# # plt.xticks(range(0, len(accs_batch_LSUV_nopp))[0:-1:5], range(0, len(accs_batch_LSUV_nopp))[0:-1:500])
# plt.title('Exp4: model accuracy per batch')
# plt.ylabel('acc')
# plt.xlabel('batch')
# plt.legend(['sgd', 'rmsp', 'adam'], loc='upper left')
# # plt.show()
# plt.savefig("exp4_acc.png")
#
#
# # loss
# fig = plt.figure(2, figsize=(40, 10))
# plt.plot(range(0, len(losses_batch_LSUV_no_dropout))[0:-1:5], losses_batch_LSUV_no_dropout[0:-1:5])
# plt.plot(range(0, len(losses_batch_LSUV_rmsp))[0:-1:5], losses_batch_LSUV_rmsp[0:-1:5])
# plt.plot(range(0, len(losses_batch_LSUV_adam))[0:-1:5], losses_batch_LSUV_adam[0:-1:5])
# # plt.xticks(range(0, len(losses_batch_LSUV_nopp))[0:-1:5], range(0, len(losses_batch_LSUV_nopp))[0:-1:5])
# plt.title('Exp4: model loss per batch')
# plt.ylabel('loss')
# plt.xlabel('batch')
# plt.legend(['sgd', 'rmsp', 'adam'], loc='upper left')
# # plt.show()
# plt.savefig("exp4_loss.png")

# plot best model
acc_path = "all_cnn_model_0.9011_0.5080.csv"
train_accs = []
test_accs = []
cnt = 0
with open(acc_path) as csvDataFile:
    csvReader = csv.reader(csvDataFile)

    for row in csvReader:
        if cnt == 0:
            cnt = cnt + 1
            continue
        train_accs.append(float(row[1]))
        test_accs.append(float(row[2]))
        cnt = cnt + 1

fig = plt.figure(2, figsize=(40, 10))
plt.plot(train_accs)
plt.plot(test_accs)
plt.title('Final model: model acc per epoch')
# plt.yticks(range(0, 100))
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['Cifar10 train', 'Cifar10 test'], loc='upper left')
# plt.show()
plt.savefig("final_model_acc.png")



from textwrap import wrap

import numpy as np
import os
import matplotlib.pyplot as plt


def load_file(result_folder, test_name, prefix=""):
    split_result = []
    # Load file
    test_res_from_file = np.loadtxt(os.path.join(result_folder, prefix + test_name + "_test"), skiprows=2)
    train_res_from_file = np.loadtxt(os.path.join(result_folder, prefix + test_name + "_train"), skiprows=2)
    valid_res_from_file = np.loadtxt(os.path.join(result_folder, prefix + test_name + "_valid"), skiprows=2)
    return train_res_from_file, test_res_from_file, valid_res_from_file


def perform_validation_loss(log_file, test_results, n_epoch, test_epoch):
    log_file.write("Validation based on loss value\n")
    print("Validation based on loss value")
    # find the best epoch in validation
    split_valid_epoch = []
    acc_split = []
    # validation on loss
    valid_res = test_results[2]

    # get loss column
    loss_list = valid_res[0:n_epoch, 1]  # NOTE:validation based on loss value, if it has to be computed on acc change index with 3, and use argmax 2 rows bellow

    # get the index (epoch) of min loss
    val_epoch_list = np.argwhere(loss_list == np.amin(loss_list))  # np.argmin(loss_list)

    # if there is many valid epoch with the same value 'ill take the best one base on acc
    acc_list = valid_res[0:n_epoch, 2]
    val_epoch = val_epoch_list[np.argmax(acc_list[val_epoch_list])]

    # get accuracy in the test ste of da validated epoch
    test_split = test_results[1]
    # acc col i the row 3 of the table
    acc = test_split[val_epoch, 2]
    acc_valid = valid_res[val_epoch, 2]

    print("Epoch: ", str(val_epoch * test_epoch), " Acc: ", acc)
    log_file.write("Epoch: " + str(val_epoch * test_epoch) + " Acc: " + str(acc_valid) + "\n")

    log_file.write("\nAcc Test: " + str(acc) + "\t Acc Valid: " + str(acc_valid) + "\n")

    return acc, acc_valid


def perform_test_eval(log_file, test_results, n_epoch, test_epoch):
    log_file.write("\n\nBest ACC in test\n")
    print("Best ACC in test")

    test_split = test_results[1]
    acc_list = test_split[0:n_epoch, 2]

    # get the index (epoch) of min loss
    best_epoch_list = np.argwhere(
        acc_list == np.amax(acc_list))  # get the list of the min (usefull if there are more epoch with the same acc

    # if there is many valid epoch with thesame value'ill take the best one base on loss
    loss_list = test_split[0:n_epoch, 1]
    best_epoch = best_epoch_list[np.argmin(loss_list[best_epoch_list])]

    # acc col i the row 3 of the table
    acc = test_split[best_epoch, 2]

    print("Epoch: ", str(best_epoch * test_epoch), " Acc: ", acc)
    log_file.write("Epoch: " + str(best_epoch * test_epoch) + " Acc: " + str(acc) + "\n")

    log_file.write("\nAcc Test: " + str(acc) + "\n")

    return acc


def perform_test_eval_PPI(log_file, test_results, n_epoch, test_epoch):
    log_file.write("\n\nBest ACC in test\n")
    print("Best ACC in test")

    test_split = test_results[1]
    acc_list = test_split[0:n_epoch, 3]

    # get the index (epoch) of min loss
    best_epoch_list = np.argwhere(
        acc_list == np.amax(acc_list))  # get the list of the min (usefull if there are more epoch with the same acc

    # if there is many valid epoch with thesame value'ill take the best one base on loss
    loss_list = test_split[0:n_epoch, 1]
    best_epoch = best_epoch_list[np.argmin(loss_list[best_epoch_list])]

    # acc col i the row 3 of the table
    acc = test_split[best_epoch, 3]

    print("Epoch: ", str(best_epoch * test_epoch), " Acc: ", acc)
    log_file.write("Epoch: " + str(best_epoch * test_epoch) + " Acc: " + str(acc) + "\n")

    log_file.write("\nAcc Test: " + str(acc) + "\n")

    return acc


def perform_validation_acc(log_file, test_results, n_epoch, test_epoch):
    log_file.write("\n\nValidation based on ACC value\n")
    print("Validation based on ACC value")

    # validation on loss

    # for i, split in enumerate(split_result):
    valid_res = test_results[2]
    # get loss column
    acc_list = valid_res[0:n_epoch, 2]

    # get the index (epoch) of max acc
    val_epoch_list = np.argwhere(acc_list == np.amax(acc_list))
    val_epoch = val_epoch_list[np.argmin(valid_res[val_epoch_list, 1])]
    # if there is many valid epoch with thesame value'ill take the first one
    # val_epoch = val_epoch_list[0]

    # get accuracy in the test ste of da validated epoch
    test_split = test_results[1]
    # acc col i the row 3 of the table
    acc = test_split[val_epoch, 2]
    acc_valid = valid_res[val_epoch, 2]

    print("Epoch: ", str(val_epoch * test_epoch), " Acc: ", acc)
    log_file.write("Epoch: " + str(val_epoch * test_epoch) + " Acc: " + str(acc_valid) + "\n")

    log_file.write("\nAcc Test: " + str(acc) + "\t Acc Valid: " + str(acc_valid) + "\n")

    return acc, acc_valid


def perform_validation_acc_PPI(log_file, test_results, n_epoch, test_epoch):
    log_file.write("\n\nValidation based on ACC value\n")
    print("Validation based on ACC value")

    # validation file
    valid_res = test_results[2]
    # get loss column
    acc_list = valid_res[0:n_epoch, 3]

    # get the index (epoch) of max acc
    val_epoch_list = np.argwhere(acc_list == np.amax(acc_list))

    # if there is many valid epoch with the same value'ill take the best one base on loss
    loss_list = valid_res[0:n_epoch, 1]
    val_epoch = val_epoch_list[np.argmin(loss_list[val_epoch_list])]

    # get accuracy in the test of the validated epoch
    test_split = test_results[1]
    # acc col i the row 3 of the table
    valid_test_acc = test_split[val_epoch, 3]
    valid_acc = valid_res[val_epoch, 3]

    print("Epoch: ", str(val_epoch * test_epoch), " Acc: ", valid_acc)
    log_file.write("Epoch: " + str(val_epoch * test_epoch) + " Acc: " + str(valid_acc) + "\n")

    log_file.write("\nAcc Test: " + str(valid_test_acc) + "\t Acc Valid: " + str(valid_acc) + "\n")

    return valid_test_acc, valid_acc


def plot_graph(test_name, result_folder, n_epoch, test_epoch, split_result, n_split, title=None):
    # evenly sampled time at 200ms intervals
    n_step = int(n_epoch / test_epoch)
    train_acc_avg = np.zeros(n_step)
    test_acc_avg = np.zeros(n_step)
    valid_acc_avg = np.zeros(n_step)

    train_loss_avg = np.zeros(n_step)
    test_loss_avg = np.zeros(n_step)
    valid_loss_avg = np.zeros(n_step)

    for i, (split_train, split_test, split_valid) in enumerate(split_result):
        # print(i)
        actual_n_epoch = split_train[:, :].shape[0]

        if actual_n_epoch < n_epoch:
            added_epoch_indexs = np.asarray(range(actual_n_epoch, n_epoch)).reshape(-1, 1)

            added_stat_value_train = np.stack([split_train[-1, 1:6]] * (n_epoch - actual_n_epoch), axis=0)
            padding_vec_train = np.concatenate((added_epoch_indexs, added_stat_value_train), axis=1)

            split_train = np.concatenate((split_train, padding_vec_train), axis=0)

            added_stat_value_test = np.stack([split_test[-1, 1:6]] * (n_epoch - actual_n_epoch), axis=0)
            padding_vec_test = np.concatenate((added_epoch_indexs, added_stat_value_test), axis=1)

            split_test = np.concatenate((split_test, padding_vec_test), axis=0)

            added_stat_value_valid = np.stack([split_valid[-1, 1:6]] * (n_epoch - actual_n_epoch), axis=0)
            padding_vec_valid = np.concatenate((added_epoch_indexs, added_stat_value_valid), axis=1)

            split_valid = np.concatenate((split_valid, padding_vec_valid), axis=0)

        n_epoch = actual_n_epoch
        train_acc_avg = split_train[0:n_epoch, 2]
        train_loss_avg = split_train[0:n_epoch, 1]

        test_acc_avg = split_test[0:n_epoch:, 2]
        test_loss_avg = split_test[0:n_epoch, 1]

        valid_acc_avg = split_valid[0:n_epoch, 2]
        valid_loss_avg = split_valid[0:n_epoch, 1]

    plt.plot(split_train[0:n_epoch, 0], train_acc_avg / n_split, 'r--', label='Train')
    plt.plot(split_test[0:n_epoch, 0], test_acc_avg / n_split, 'b--', label="Test")
    plt.plot(split_valid[0:n_epoch, 0], valid_acc_avg / n_split, 'g--', label="Valid")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    if title is None:
        plt.title("\n".join(wrap(test_name.replace("_", " "))))
    else:
        plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(result_folder, test_name + "_ACC.pdf"))
    # plt.show()
    plt.clf()

    # plt.ylim(bottom=0, top=2)
    plt.plot(split_train[0:n_epoch, 0], train_loss_avg / n_split, 'r--', label='Train')
    plt.plot(split_test[0:n_epoch, 0], test_loss_avg / n_split, 'b--', label='Test')
    plt.plot(split_valid[0:n_epoch, 0], valid_loss_avg / n_split, 'g--', label='Valid')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if title is None:
        plt.title("\n".join(wrap(test_name.replace("_", " "))))
    else:
        plt.title(title)
    plt.legend()
    plt.savefig(os.path.join(result_folder, test_name + "_LOSS.pdf"))
    plt.clf()
    # plt.show()
    # pass


def perform_time_eval(log_file, test_results, n_epoch, test_epoch):
    log_file.write("\n\nBest ACC in test\n")
    print("Best ACC in test")

    train_split = test_results[0]
    time_list = train_split[0:n_epoch, 3]

    train_epoch_avg_time = np.asarray(time_list).mean()

    print("avg_time per epoch: ", str(train_epoch_avg_time))
    log_file.write("avg_time per epoch: " + str(train_epoch_avg_time) + "\n")

    return train_epoch_avg_time

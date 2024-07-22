import os
import sys
import numpy as np
from validation.results_parser_utils import load_file, perform_validation_acc, perform_test_eval, plot_graph
from validation.results_parser_utils import perform_validation_acc_PPI, perform_test_eval_PPI

sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

script_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(script_dir, '..'))
sys.path.append(parent_dir)

if __name__ == '__main__':
    test_title = "HLGNN"
    test_folder = os.path.join(parent_dir, "experiments/citeseer/test_log/LGCN-fully-supervised/data")

    n_epoch = 200
    test_epoch = 5
    max_run = 5
    file_init = True

    # ----------------------------#

    test_list = os.listdir(test_folder)
    if ".directory" in test_list:
        test_list.remove(".directory")
    if "results" in test_list:
        test_list.remove("results")
    if "best_test.txt" in test_list:
        test_list.remove("best_test.txt")
    if "exp_summary.txt" in test_list:
        test_list.remove("exp_summary.txt")
    if ".DS_Store" in test_list:
        test_list.remove(".DS_Store")

    results_folder = os.path.join(test_folder, "results")
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)

    overall_test_log = open(os.path.join(results_folder, "results.log"), 'w+')

    test_instance = set([])
    for test_name in test_list:

        instance_name = test_name[6:]
        # instance_name = test_name[8:] #TODO
        if instance_name[0] == '_':
            instance_name = instance_name[1:]

        test_instance.add(instance_name)

    print("test instance", test_instance)

    for test in test_instance:
        test_runs_results_test = []
        test_runs_results_valid = []

        test_runs_results_no_valid = []

        for run in range(max_run):
            # current_run = "re_run_" + str(run) + "_" + test  #TODO
            current_run = "run_" + str(run) + "_" + test  # TODO
            print("TEST: ", current_run)
            current_test_folder = os.path.join(test_folder, current_run)
            if os.path.exists(current_test_folder):
                test_log_file_o = open(os.path.join(current_test_folder, current_run + ".log"), 'r')
                test_log_file = test_log_file_o.readlines()[3:]
                test_log_file_o.close()
                test_par = dict()
                for par in test_log_file:
                    if "Validation results" in par:
                        break
                    name, value = par[0: par.find('\t') - 1], par[par.find('\t') + 1:]
                    test_par[name.replace(':', '')] = value.replace('\n', '')

                test_results = load_file(current_test_folder, current_run)

                if test_results is None or len(test_results[0]) == 0 or len(test_results[1]) == 0 or len(
                        test_results[2]) == 0 \
                        or len(test_results[0].shape) < 2 or len(test_results[1].shape) < 2 or \
                        len(test_results[2].shape) < 2:
                    continue

                test_log_file_o = open(os.path.join(current_test_folder, current_run + ".log"), 'a')
                test_log_file_o.write("\nValidation results\n")
                test_log_file_o.write("\n")

                # acc, acc_valid = perform_validation_acc_PPI(test_log_file_o, test_results, n_epoch, test_epoch)
                acc, acc_valid = perform_validation_acc(test_log_file_o, test_results, n_epoch, test_epoch)
                # acc_test = perform_test_eval_PPI(test_log_file_o, test_results, n_epoch, test_epoch)
                acc_test = perform_test_eval(test_log_file_o, test_results, n_epoch, test_epoch)

                test_runs_results_test.append(acc)
                test_runs_results_valid.append(acc_valid)

                test_runs_results_no_valid.append(acc_test)

                plot_graph(current_run, results_folder, n_epoch, test_epoch, [test_results], 1,
                           title=test_title + " - " + test_par["dataset_name"].capitalize() + " Dataset")
            else:
                print(os.path.join(test_folder, current_run), " not exists")

        if file_init:
            file_init = False
            overall_test_log.write(
                "dataset_name\ttest_type\trun\tdrop_out\tweight_decay\tnum_layers\thidden_dim\tbacktrack\tactivation\tinc_type\tupdate\tacc test\t test stdev\tacc valid\tvalid stdev\tacc_no_valid\tno_valid_std\n")
            # overall_test_log.write(
            #     "dataset_name\ttest_type\trun\tk\tbeta\tn_hidden\tdrop_out\tlearning_rate\tweight_decay\tacc test\t test stdev\tacc valid\tvalid stdev\tacc_no_valid\tno_valid_std\n")#TODO: verisone con beta

        overall_test_log.write(test_par["dataset_name"])
        overall_test_log.write("\t")

        overall_test_log.write(test[0:test.find("_data")])
        overall_test_log.write("\t")

        overall_test_log.write(str(len(test_runs_results_test)))
        overall_test_log.write("\t")

        overall_test_log.write(test_par["dropout"])
        overall_test_log.write("\t")

        overall_test_log.write(test_par["weight_decay"])
        overall_test_log.write("\t")

        overall_test_log.write(test_par["num_layers"])
        overall_test_log.write("\t")

        overall_test_log.write(test_par["hidden_dim"])
        overall_test_log.write("\t")

        overall_test_log.write(test_par["backtrack"])
        overall_test_log.write("\t")

        if 'activation' in test_par:
            overall_test_log.write(test_par["activation"])
            overall_test_log.write("\t")
        else:
            overall_test_log.write('ReLU')
            overall_test_log.write("\t")

        if 'inc_type' in test_par:
            overall_test_log.write(test_par["inc_type"])
            overall_test_log.write("\t")
        else:
            overall_test_log.write('in')
            overall_test_log.write("\t")

        if 'update' in test_par:
            overall_test_log.write(test_par["update"])
            overall_test_log.write("\t")
        else:
            overall_test_log.write('sum')
            overall_test_log.write("\t")

        overall_test_log.write(str(np.mean(np.asarray(test_runs_results_test))) + '\t' + str(
            np.std(np.asarray(test_runs_results_test))) + '\t' + str(
            np.mean(np.asarray(test_runs_results_valid))) + '\t' + str(
            np.std(np.asarray(test_runs_results_valid))) + '\t' + str(
            np.mean(np.asarray(test_runs_results_no_valid))) + '\t' + str(
            np.std(np.asarray(test_runs_results_no_valid))) + '\n')

        # for key, value in test_par.items():
        #     overall_test_log.write(value + '\t')

        # write

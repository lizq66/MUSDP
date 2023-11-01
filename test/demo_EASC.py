import multiprocessing
import os
import warnings
import numpy as np
import optuna
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from algorithms.Classifier_doer import CLF_paramers
from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


def tune_clf(train_data, train_label, randseed):
    train_data = np.c_[train_data, train_label]

    train_data, train_label, val_data, val_label, _, _ = outofsample_bootstrap(pd.DataFrame(train_data), randseed)
    optimizer = CLF_paramers(train_data, train_label, val_data, val_label, 'NB')

    study = optuna.create_study(study_name='classifier_tuning', load_if_exists=False,
                                directions=['maximize'], sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: optimizer.optCLF_fun(trial), n_trials=50, n_jobs=-1)

    clf = optimizer.getCLF(study.best_params)

    return clf


def multi_one_run_(Xt, Xs, save_path, project_name, model_name, randseed):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')

    Xtr, Xtr_y, Xte, Xte_y, _, _ = outofsample_bootstrap(Xt, randseed)
    LOC = Xte['CountLineCode']

    Xsrc = Xs.iloc[:, :-1]
    Xsrc_y = Xs.iloc[:, -1]
    Xsrc_y[Xsrc_y > 1] = 1

    # log transformation
    Xsrc = np.log(Xsrc + 1)
    Xte = np.log(Xte + 1)

    # build model
    # clfmodel = GaussianNB()
    clfmodel = tune_clf(Xsrc, Xsrc_y, randseed)
    clfmodel.fit(Xsrc, Xsrc_y)
    predict_y = clfmodel.predict(Xte)

    # # calculate non-effort-aware classification measure
    AUC, MCC = performanceMeasure.get_measure(Xte_y, predict_y)
    # # calculate cost-effectiveness measures
    Popt, CostEffort = rankMeasure.rank_measure(predict_y, LOC, Xte_y)
    measure = [AUC, MCC, Popt, CostEffort]

    fres = create_dir(save_path + model_name)
    save_results(fres + project_name, measure)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    save_path = '../result/'
    Reps = 100
    model_name = 'EASC'
    max_cpu = multiprocessing.cpu_count()

    project_names = sorted(os.listdir('../data/'))
    path = os.path.abspath('../data/')
    pro_num = len(project_names)
    for i in range(pro_num):
        project_name = project_names[i]
        file = os.path.join(path, project_name)
        tar_data = pd.read_csv(file)
        tar_project_name = project_name[:-4]

        temp_data = []
        for j in range(pro_num):
            if i == j:
                continue

            project_name = project_names[j]
            src_project_name = project_name[:-4]
            if tar_project_name.split("-")[0] == src_project_name.split("-")[0]:
                continue  # different releases of the same project

            file = os.path.join(path, project_name)
            data = pd.read_csv(file)
            temp_data.append(data)

            # combining one pd.DataFrame
            src_data = pd.concat(temp_data, ignore_index=True)

        for loop in range(Reps):
            multi_one_run_(tar_data, src_data, save_path, tar_project_name, model_name, loop)

    print('done!')

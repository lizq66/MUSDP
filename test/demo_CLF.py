import os
import warnings
import numpy as np
import optuna
import pandas as pd
from sklearn import preprocessing
from algorithms.Classifier_doer import CLF_paramers
from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X, save_path, project_name, model_name, randseed):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')

    train_data, train_label, test_data, test_label, _, _ = outofsample_bootstrap(X, randseed)
    LOC = test_data['CountLineCode']
    test_data = preprocessing.scale(test_data)

    # classifier
    train_data = np.c_[train_data, train_label]

    train_data, train_label, val_data, val_label, _, _ = outofsample_bootstrap(pd.DataFrame(train_data), randseed)
    train_data = preprocessing.scale(train_data)
    val_data = preprocessing.scale(val_data)

    optimizer = CLF_paramers(train_data, train_label, val_data, val_label, model_name)

    study = optuna.create_study(study_name='classifier_tuning', load_if_exists=False,
                                directions=['maximize'],
                                sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: optimizer.optCLF_fun(trial), n_trials=50, n_jobs=-1)

    clf = optimizer.getCLF(study.best_params)
    clf.fit(train_data, train_label)
    predict_y = clf.predict(test_data)

    # # calculate non-effort-aware classification measure
    AUC, MCC = performanceMeasure.get_measure(test_label, predict_y)
    # # calculate cost-effectiveness measures
    Popt, CostEffort = rankMeasure.rank_measure(predict_y, LOC, test_label)
    measure = [AUC, MCC, Popt, CostEffort]

    fres = create_dir(save_path + model_name)
    save_results(fres + project_name, measure)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    save_path = '../result/'
    CLF = ['LR', 'RF']
    Reps = 100

    project_names = sorted(os.listdir('../data/'))
    path = os.path.abspath('../data/')
    pro_num = len(project_names)
    for model_name in CLF:
        for i in range(0, pro_num):
            project_name = project_names[i]
            file = os.path.join(path, project_name)
            data = pd.read_csv(file)
            project_name = project_name[:-4]

            for loop in range(Reps):
                run_(data, save_path, project_name, model_name, loop)

    print('done!')

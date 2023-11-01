import os
import warnings
import numpy as np
import optuna
import pandas as pd
import torch
from sklearn import preprocessing
from algorithms.DNN.DNN import DeepNN
from algorithms.DNN.loadDataset import MyDataset
from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X, save_path, project_name, model_name, args, randseed, tag='opt'):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')

    train_data, train_label, test_data, test_label, _, _ = outofsample_bootstrap(X, randseed)
    LOC = test_data['CountLineCode']
    test_data = preprocessing.scale(test_data)
    test_label = np.asarray(test_label)

    if tag == 'default':
        train_data = preprocessing.scale(train_data)
        train_label = np.asarray(train_label)
    else:
        # split training and validation data
        train_data = np.c_[train_data, train_label]
        # train_data = np.unique(train_data, axis=0)  # delete repetitive rows

        train_data, train_label, val_data, val_label, _, _ = outofsample_bootstrap(pd.DataFrame(train_data), randseed)
        train_data = preprocessing.scale(train_data)
        train_label = np.asarray(train_label)
        val_data = preprocessing.scale(val_data)
        val_label = np.asarray(val_label)

        # deep learning model
        dlmodel = DeepNN(train_data, train_label, val_data, val_label, args).to(args['device'])

        study = optuna.create_study(study_name='classifier_tuning', load_if_exists=False,
                                    directions=['maximize'], sampler=optuna.samplers.TPESampler())
        study.optimize(lambda trial: dlmodel.objective(trial), n_trials=30, n_jobs=-1)

        args.update(study.best_params)

    train_loader = torch.utils.data.DataLoader(dataset=MyDataset(train_data, train_label), batch_size=args['batch_size'],
                                            shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=MyDataset(test_data, test_label), batch_size=args['batch_size'])

    dlmodel = DeepNN(train_data, train_label, test_data, test_label, args).to(args['device'])
    dlmodel.train_dl(dlmodel, train_loader)
    predict_y, true_y = dlmodel.predict_dl(dlmodel, test_loader)

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

    save_path = "../result/"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = 'DNN'
    args = {'epochs': 100, 'batch_size': 512, 'in_dim': 65, 'out_dim': 2, 'device': device,
            'lr': 1e-3, 'n_units': 100, 'n_layers': 1,  'dropout': 0.15}
    Reps = 100

    project_names = sorted(os.listdir('../data/'))
    path = os.path.abspath('../data/')
    pro_num = len(project_names)
    for i in range(0, pro_num):
        project_name = project_names[i]
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)
        project_name = project_name[:-4]

        for loop in range(Reps):
            run_(data, save_path, project_name, model_name, args, loop, tag='opt')

    print('done!')

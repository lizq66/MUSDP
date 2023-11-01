import numpy as np
import optuna
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from algorithms.Classifier_doer import CLF_paramers
from utilities.bootstrapCV import outofsample_bootstrap

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune_clf(train_data, train_label, randseed, model_name):
    train_data = np.c_[train_data, train_label]

    train_data, train_label, val_data, val_label, _, _ = outofsample_bootstrap(pd.DataFrame(train_data), randseed)
    optimizer = CLF_paramers(train_data, train_label, val_data, val_label, model_name)

    study = optuna.create_study(study_name='classifier_tuning', load_if_exists=False,
                                directions=['maximize'], sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: optimizer.optCLF_fun(trial), n_trials=50, n_jobs=-1)

    clf = optimizer.getCLF(study.best_params)

    return clf


class Mvknn(object):
    def __init__(self, trainx, trainy, testx, testy, randseed):
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy
        self.randseed = randseed

    def predict(self):
        # neighbors = np.ceil(np.sqrt(len(self.trainy))).astype(np.int64)
        y_mvknn = []
        for i in range(len(self.trainx)):
            trx = self.trainx[i]
            tex = self.testx[i]
            pred_y = []
            # for k in range(neighbors):
            #     clf = KNeighborsClassifier(n_neighbors=k+1, n_jobs=-1)
            #     clf.fit(trx, self.trainy)
            #     y = clf.predict(tex)
            #     pred_y.append(y)
            # pred_y = np.mean(pred_y, axis=0)
            # y_mvknn.append(pred_y)

            clf = tune_clf(trx, self.trainy, self.randseed, 'KNN')
            clf.fit(trx, self.trainy)
            pred_y = clf.predict(tex)
            y_mvknn.append(pred_y)

        # majority voting
        y_mvknn = np.mean(y_mvknn, axis=0)

        y_mvknn[y_mvknn >= 0.5] = 1
        y_mvknn[y_mvknn < 0.5] = 0

        return y_mvknn
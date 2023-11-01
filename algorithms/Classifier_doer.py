import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import matthews_corrcoef
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier


class CLF_paramers(object):
    def __init__(self, trainx, trainy, testx, testy, classifier):
        self.trainx = trainx
        self.trainy = trainy
        self.testx = testx
        self.testy = testy
        self.clf = classifier

    def run(self):
        if self.clf == 'LR':
            clfmodel = LogisticRegression(solver='liblinear', n_jobs=-1)

        if self.clf == 'RF':
            clfmodel = RandomForestClassifier(n_jobs=-1)

        if self.clf == 'NB':
            clfmodel = GaussianNB()

        if self.clf == 'KNN':
            clfmodel = KNeighborsClassifier(n_jobs=-1)

        # training model
        clfmodel.fit(self.trainx, self.trainy)

        # prediction
        y_pred = clfmodel.predict(self.testx)

        return y_pred

    def optCLF_fun(self, trial):
        if self.clf == 'LR':
            penalty = trial.suggest_categorical('penalty', ['l1', 'l2'])
            C = trial.suggest_float('C', 0.0001, 1000)
            tol = trial.suggest_float('tol', 0.00001, 1)
            clfmodel = LogisticRegression(penalty=penalty, C=C, tol=tol, solver='liblinear', n_jobs=-1)

        if self.clf == 'RF':
            n_estimators = trial.suggest_int('n_estimators', 10, 500)
            max_depth = trial.suggest_int('max_depth', 1, 20)
            criterion = trial.suggest_categorical('criterion', ['gini', 'entropy']) #
            clfmodel = RandomForestClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth,
                                              criterion=criterion, n_jobs=-1)

        if self.clf == 'NB': # for tuning EASC
            NBType = trial.suggest_categorical('NBType', ['GaussianNB', 'MultinomialNB', 'BernoulliNB'])
            if NBType == 'GaussianNB':
                clfmodel = GaussianNB()
            elif NBType == 'MultinomialNB':
                clfmodel = MultinomialNB()
            elif NBType == 'BernoulliNB':
                clfmodel = BernoulliNB()

        if self.clf == 'KNN': # for tuning MVKNN
            # MVKNN: neighbors from 1 to sqrt(n)
            n = np.ceil(np.sqrt(len(self.trainy)))
            n_neighbors = trial.suggest_int('n_neighbors', 1, n)
            weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
            p = trial.suggest_int('p', 1, 5)
            clfmodel = KNeighborsClassifier(n_neighbors=n_neighbors,
                                            weights=weights,
                                            p=p, n_jobs=-1)

        # training model
        clfmodel.fit(self.trainx, self.trainy)

        # predict
        y_pred = clfmodel.predict(self.testx)

        return matthews_corrcoef(self.testy, y_pred)

    def getCLF(self, params):
        if self.clf == 'LR':
            return LogisticRegression(**params, solver='liblinear', n_jobs=-1)

        if self.clf == 'RF':
            return RandomForestClassifier(**params, n_jobs=-1)

        if self.clf == 'NB':
            if params['NBType'] == 'GaussianNB':
                return GaussianNB()
            elif params['NBType'] == 'MultinomialNB':
                return MultinomialNB()
            elif params['NBType'] == 'BernoulliNB':
                return BernoulliNB()

        if self.clf == 'KNN':
            return KNeighborsClassifier(**params, n_jobs=-1)

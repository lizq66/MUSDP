import os
import warnings
import pandas as pd
from sklearn import preprocessing
from algorithms.MVKNN.MVKNN import Mvknn
from utilities import performanceMeasure, rankMeasure
from utilities.File import create_dir, save_results
from utilities.bootstrapCV import outofsample_bootstrap


def run_(X, save_path, project_name, model_name, randseed):
    print(project_name + ': -> ' + model_name + ' ' + str(randseed + 1) + ' round Start!')

    train_data, train_label, test_data, test_label, _, _ = outofsample_bootstrap(X, randseed)
    LOC = test_data[0]['CountLineCode']

    train_data = [preprocessing.scale(x) for x in train_data]
    test_data = [preprocessing.scale(x) for x in test_data]

    # MVKNN
    mvknn = Mvknn(train_data, train_label, test_data, test_label, randseed)
    predict_y = mvknn.predict()

    # # calculate non-effort-aware classification measure
    AUC, MCC = performanceMeasure.get_measure(test_label, predict_y)
    # # calculate cost-effectiveness measures
    Popt, CostEffort = rankMeasure.rank_measure(predict_y, LOC, test_label)
    measure = [AUC, MCC, Popt, CostEffort]

    fres = create_dir(save_path + model_name)
    save_results(fres + project_name, measure)


if __name__ == '__main__':
    warnings.filterwarnings('ignore')

    save_path = '../result/'
    Reps = 100
    model_name = 'MVKNN'

    project_names = sorted(os.listdir('../data/'))
    path = os.path.abspath('../data/')
    pro_num = len(project_names)

    for i in range(0, pro_num):
        project_name = project_names[i]
        file = os.path.join(path, project_name)
        data = pd.read_csv(file)
        project_name = project_name[:-4]

        # construct multi-view data
        X = [data.iloc[:, 0:54], data.iloc[:, 54:59], data.iloc[:, 59:65]]  #
        y = data.iloc[:, -1]
        X.append(y)

        for loop in range(Reps):
            run_(X, save_path, project_name, model_name, loop)

    print('done!')

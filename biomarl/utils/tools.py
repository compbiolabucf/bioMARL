import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeClassifier, Ridge, Lasso, LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import LinearSVC, LinearSVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier, XGBRegressor
from biomarl.utils.logger import info
from sklearn.calibration import CalibratedClassifierCV


def relative_absolute_error(y_test, y_predict):
    y_test = np.array(y_test)
    y_predict = np.array(y_predict)
    error = np.sum(np.abs(y_test - y_predict)) / np.sum(np.abs(np.mean(
        y_test) - y_test))
    return error


def downstream_task_new(data, task_type):
    print('inside downstream task new now')
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    # if task_type == 'cls':
    #     clf = RandomForestClassifier(random_state=42, n_jobs=128)
    #     f1_list = []
    #     skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
    #     for train, test in skf.split(X, y):
    #         X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
    #         ], X.iloc[test, :], y.iloc[test]
    #         clf.fit(X_train, y_train)
    #         y_predict = clf.predict(X_test)
    #         f1_list.append(f1_score(y_test, y_predict, average='weighted'))
    #     return np.mean(f1_list)
    if task_type == 'cls':
        clf = RandomForestClassifier(n_jobs = 128)
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        # skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        # skf = StratifiedShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
        skf = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
        for train, test in skf.split(X, y):
            print(f'Test indices: {test}')
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            # y_predict = clf.predict(X_test)
            y_predict = clf.predict(X_test)
            y_predict_prob = clf.predict_proba(X_test)[:,1]
            # y_predict = np.copy(y_predict_prob)
            # y_predict[y_predict < 0.5] = 0
            # y_predict[y_predict >= 0.5] = 1
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            auc_roc_score.append(roc_auc_score(y_test, y_predict_prob, average='weighted'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0, n_jobs=128)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'det':
        knn = KNeighborsClassifier(n_neighbors=5, n_jobs=128)
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        ras_list = []
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            ras_list.append(roc_auc_score(y_test, y_predict))
        return np.mean(ras_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0, n_jobs=128))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    elif task_type == 'rank':
        pass
    else:
        return -1

# 'RF', 'XGB', 'SVM', 'KNN', 'Ridge'
def downstream_task_by_method(data, task_type, method='RF'):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso()
    else:  # dt
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list)
    else:
        return -1


def downstream_task_by_method_std(data, task_type, method='RF'):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1].astype(float)
    if method == 'RF':
        if task_type == 'cls':
            model = RandomForestClassifier(random_state=0, n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RandomForestClassifier(random_state=0), n_jobs=128)
        else:
            model = RandomForestRegressor(random_state=0, n_jobs=128)
    elif method == 'XGB':
        if task_type == 'cls':
            model = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            model = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task_type == 'cls':
            model = LinearSVC()
        elif task_type == 'mcls':
            model = LinearSVC()
        else:
            model = LinearSVR()
    elif method == 'KNN':
        if task_type == 'cls':
            model = KNeighborsClassifier(n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            model = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task_type == 'cls':
            model = RidgeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            model = Ridge()
    elif method == 'LASSO':
        if task_type == 'cls':
            model = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task_type == 'mcls':
            model = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            model = Lasso()
    else:  # dt
        if task_type == 'cls':
            model = DecisionTreeClassifier()
        elif task_type == 'mcls':
            model = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            model = DecisionTreeRegressor()

    if task_type == 'cls':
        f1_list = []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
        return np.mean(f1_list), np.std(f1_list)
    elif task_type == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        rae_list = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train
            ], X.iloc[test, :], y.iloc[test]
            model.fit(X_train, y_train)
            y_predict = model.predict(X_test)
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
        return np.mean(rae_list), np.std(rae_list)
    elif task_type == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
        return np.mean(f1_list), np.std(f1_list)
    else:
        return -1



def test_task_wo_cv(Dg, task='cls'):
    X = Dg.iloc[:, :-1]
    y = Dg.iloc[:, -1].astype(float)
    if task == 'cls':
        clf = RandomForestClassifier(random_state=0)
        pre_list, rec_list, f1_list, auc_roc_score = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            auc_roc_score.append(roc_auc_score(y_test, y_predict, average='weighted'))
            break
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list, rmse_list = [], [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            reg.fit(X_train, y_train)
            y_predict = reg.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict, squared=True))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
            break
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        recall = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            ras.append(roc_auc_score(y_test, y_predict))
            recall.append(recall_score(y_test, y_predict, average='weighted'))
            break
        return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    elif task == 'mcls':
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, maf1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='macro'))
            rec_list.append(recall_score(y_test, y_predict, average='macro'))
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
            maf1_list.append(f1_score(y_test, y_predict, average='macro'))
            break
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    elif task == 'rank':
        pass
    else:
        return -1
    
def test_task_et(train_data, test_data, task='cls'):
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:,-1].astype(float)
    X_test = test_data.iloc[:, :-1]
    y_test = test_data.iloc[:, -1].astype(float)

    if task == 'cls':
        print('yes we are here now inside ehtesams code')
        print(f'shape of X_train is: {X_train.shape}')
        print(f'shape of x_test is: {X_test.shape}')
        print(f'Test indices: {sorted(y_test.index)}')
        print(f'Test target values: {y_test.values}')
        print(f'length of test indices: {len(y_test)}')
        clf = RandomForestClassifier(random_state=0)
        clf.fit(X_train, y_train)
        y_predict = clf.predict(X_test)
        precision = precision_score(y_test, y_predict, average='weighted')
        recall = recall_score(y_test, y_predict, average='weighted')
        f1 = f1_score(y_test, y_predict, average='weighted')
        auc_roc = roc_auc_score(y_test, y_predict, average='weighted')
        return precision, recall, f1, auc_roc
    else:
        return -1 #because this is only for the two smaller datasets, if you have enough sample please use test_task_new


def test_task_new(Dg, task='cls', method='RF'):
    # print('in test task wdj now')
    X = Dg.iloc[:, :-1].astype(float)
    y = Dg.iloc[:, -1].astype(float)
    if method == 'RF':
        if task == 'cls':
            # clf = RandomForestClassifier(n_jobs=128, n_estimators=200, random_state=42, min_samples_leaf=5,  min_samples_split=10, max_features='sqrt')
            clf = RandomForestClassifier(n_jobs=128,
                                        n_estimators=100,
                                        random_state=0,
                                        min_samples_leaf=10,
                                        min_samples_split=20, 
                                        max_features=0.5, 
                                        max_depth=10, 
                                        bootstrap=True
                                    )

        elif task == 'mcls':
            clf = OneVsRestClassifier(RandomForestClassifier, n_jobs=128)
        else:
            clf = RandomForestRegressor(n_jobs=128)
    elif method == 'XGB':
        if task == 'cls':
            clf = XGBClassifier(eval_metric='logloss', n_jobs=128)
        elif task == 'mcls':
            clf = OneVsRestClassifier(XGBClassifier(eval_metric='logloss'), n_jobs=128)
        else:
            clf = XGBRegressor(eval_metric='logloss', n_jobs=128)
    elif method == 'SVM':
        if task == 'cls':
            base_clf = LinearSVC()
            clf = CalibratedClassifierCV(base_clf)
        elif task == 'mcls':
            base_clf = LinearSVC()
            clf = OneVsRestClassifier(CalibratedClassifierCV(base_clf), n_jobs=128)
        else:
            clf = LinearSVR()
    elif method == 'KNN':
        if task == 'cls':
            clf = KNeighborsClassifier(n_jobs=128)
        elif task == 'mcls':
            clf = OneVsRestClassifier(KNeighborsClassifier(), n_jobs=128)
        else:
            clf = KNeighborsRegressor(n_jobs=128)
    elif method == 'Ridge':
        if task == 'cls':
            clf = RidgeClassifier()
        elif task == 'mcls':
            clf = OneVsRestClassifier(RidgeClassifier(), n_jobs=128)
        else:
            clf = Ridge()
    elif method == 'LASSO':
        if task == 'cls':
            clf = LogisticRegression(penalty='l1',solver='liblinear', n_jobs=128)
        elif task == 'mcls':
            clf = OneVsRestClassifier(LogisticRegression(penalty='l1',solver='liblinear'), n_jobs=128)
        else:
            clf = Lasso()
    else:  # dt
        if task == 'cls':
            clf = DecisionTreeClassifier()
        elif task == 'mcls':
            clf = OneVsRestClassifier(DecisionTreeClassifier(), n_jobs=128)
        else:
            clf = DecisionTreeRegressor()
    # print(f'shape of X is: {X.shape}')
    if task == 'cls':
        # clf = RandomForestClassifier(n_jobs = 128)
        # info(f'fitting in progress with: {clf}')
        pre_list, rec_list, f1_list, auc_roc_score, pr_auc_score = [], [], [], [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        all_test_indices = set()
        # skf = StratifiedShuffleSplit(n_splits=5, test_size=0.3)
        # skf = StratifiedShuffleSplit(n_splits=5, test_size=0.2)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            # print(f'shape of X_train is: {X_train.shape}')
            # print(f'shape of x_test is: {X_test.shape}')
            # print(f'Test indices: {sorted(y_test.index)[:10]}')

            # print(f'Test target values: {y_test.values}')
            # print(f'length of test indices: {len(test)}')
            # train_ratio = y_train.value_counts(normalize=True)
            # test_ratio = y_test.value_counts(normalize=True)
            # print(f'Train ratio: {train_ratio.to_dict()}')
            # print(f'Test ratio: {test_ratio.to_dict()}')

            # all_test_indices.update(y_test.index)

            # Check if all indices are covered
            # if len(all_test_indices) == len(Dg):
            #     print("All data points are included in the test sets across splits.")
            # else:
            #     print("Some data points are missing in the test sets.")
            clf.fit(X_train, y_train)
            # y_predict = clf.predict(X_test)
            y_predict = clf.predict(X_test)
            if hasattr(clf, "predict_proba"):
                y_predict_prob = clf.predict_proba(X_test)[:, 1]
            else:
                # For SVM, use decision_function instead
                y_predict_prob = clf.decision_function(X_test)
            # y_predict_prob = clf.predict_proba(X_test)[:,1]
            pre_list.append(precision_score(y_test, y_predict, average='weighted'))
            rec_list.append(recall_score(y_test, y_predict, average='weighted'))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            auc_roc_score.append(roc_auc_score(y_test, y_predict_prob))
            pr_auc_score.append(average_precision_score(y_test, y_predict_prob))
        # print(f'list of roc auc scores is: {auc_roc_score}')
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(auc_roc_score), np.mean(pr_auc_score)
    elif task == 'reg':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        # reg = RandomForestRegressor(random_state=0)
        mae_list, mse_list, rae_list, rmse_list = [], [], [], []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            mae_list.append(1 - mean_absolute_error(y_test, y_predict))
            mse_list.append(1 - mean_squared_error(y_test, y_predict, squared=True))
            rae_list.append(1 - relative_absolute_error(y_test, y_predict))
            rmse_list.append(1 - mean_squared_error(y_test, y_predict, squared=False))
        return np.mean(mae_list), np.mean(mse_list), np.mean(rae_list), np.mean(rmse_list)
    elif task == 'det':
        kf = KFold(n_splits=5, random_state=0, shuffle=True)
        knn_model = KNeighborsClassifier(n_neighbors=5)
        map_list = []
        f1_list = []
        ras = []
        recall = []
        for train, test in kf.split(X):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            knn_model.fit(X_train, y_train)
            y_predict = knn_model.predict(X_test)
            map_list.append(average_precision_score(y_test, y_predict))
            f1_list.append(f1_score(y_test, y_predict, average='weighted'))
            ras.append(roc_auc_score(y_test, y_predict))
            recall.append(recall_score(y_test, y_predict, average='weighted'))
        return np.mean(map_list), np.mean(f1_list), np.mean(ras), np.mean(recall)
    elif task == 'mcls':
        # print('YES WE DOING MULTICLASS BABYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY')
        clf = OneVsRestClassifier(RandomForestClassifier(random_state=0))
        pre_list, rec_list, f1_list, maf1_list = [], [], [], []
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        for train, test in skf.split(X, y):
            X_train, y_train, X_test, y_test = X.iloc[train, :], y.iloc[train], X.iloc[test, :], y.iloc[test]
            # print(f'shape of X_train is: {X_train.shape}')
            # print(f'shape of x_test is: {X_test.shape}')
            # print(f'Test indices: {sorted(y_test.index)[:10]}')
            # print(f'Test target values: {y_test.values}')
            # print(f'length of test indices: {len(test)}')
            # train_ratio = y_train.value_counts(normalize=True)
            # test_ratio = y_test.value_counts(normalize=True)
            # print(f'Train ratio: {train_ratio.to_dict()}')
            # print(f'Test ratio: {test_ratio.to_dict()}')
            clf.fit(X_train, y_train)
            y_predict = clf.predict(X_test)
            pre_list.append(precision_score(y_test, y_predict, average='macro'))
            rec_list.append(recall_score(y_test, y_predict, average='macro'))
            f1_list.append(f1_score(y_test, y_predict, average='micro'))
            maf1_list.append(f1_score(y_test, y_predict, average='macro'))
        return np.mean(pre_list), np.mean(rec_list), np.mean(f1_list), np.mean(maf1_list)
    elif task == 'rank':
        pass
    else:
        return -1

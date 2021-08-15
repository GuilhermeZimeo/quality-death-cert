"""Create features, split data and fit predictive models."""
import os
import joblib
import argparse
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from utils.do_nothing import DoNothing


def main(args=None):
    """Run all methods available."""
    # let user choose target and group variable for fitting the model
    if args is None:
        parser = argparse.ArgumentParser()

        parser.add_argument(
            '-t', '--target',
            type=str,
            default='data',
            help='target variable: choose between "data" or "gbd"')

        parser.add_argument(
            '-g', '--group',
            type=str,
            default='regsaude',
            help='group variable: choose between "regsaude" or "uf"')

        args = parser.parse_args()

    # read merged data, create features and split them in different sets
    df = pd.read_csv(os.path.join('data', 'preproc', 'merged_data.csv'))
    X, y, groups = get_features(df, target=args.target, group=args.group)
    X_train, _, _, y_train, _, _, groups_train, _, _ = \
        split_features(X, y, groups)

    # get param_grid, fit data to grid search and export best estimator
    param_grid = get_param_grid()
    _ = export_best_model(param_grid, X_train, y_train, groups_train)


def get_features(df, target='data', group='regsaude'):
    """Get features (X, y, groups) from merged data."""
    # translate target input to available cols
    if target == 'data':
        target_col = 'Data Quality (%)'
    elif target == 'gbd':
        target_col = 'GBD Quality (%)'
    else:
        raise ValueError(f'Input "{target}" not recognized.')

    # translate group input to available cols
    if group == 'regsaude':
        group_col = group
    elif group == 'uf':
        group_col = 'UF_x'
    else:
        raise ValueError(f'Input "{group}" not recognized.')

    cols_to_keep = [
        # GeoSES
        'Education',
        'Poverty',
        'Wealth',
        'Income',
        'Deprivation',

        # CNES
        'P_DOCTORS',
        'P_HOSPBEDS',
        'VINC_SUS',
        'URGEMERG',

        # SIM
        'DEATH_RATE',

        # target variable
        target_col,

        # groups
        group_col
    ]

    filt_df = df[cols_to_keep].dropna()
    X = filt_df.drop([target_col, group_col], axis=1).copy()

    thres = filt_df[target_col].median()
    y = filt_df[target_col].apply(lambda x: 1 if x > thres else 0).copy()

    groups = filt_df[group_col].copy()

    pathout = os.path.join('data', 'features')
    if not os.path.exists(pathout):
        os.makedirs(pathout)
    X.to_csv(os.path.join(pathout, 'X.csv'), index=False)
    y.to_csv(os.path.join(pathout, 'y.csv'), index=False)
    groups.to_csv(os.path.join(pathout, 'groups.csv'), index=False)

    return X, y, groups


def split_features(X, y, groups):
    """Split features in train, validation and test sets (based on groups)."""
    # firstly, split 70% training and 30% test
    train_inds, test_inds = next(GroupShuffleSplit(
        test_size=0.3, random_state=0).split(X, y, groups=groups))
    X_train, X_test, y_train, y_test = \
        X.iloc[train_inds, :], X.iloc[test_inds, :], \
        y.iloc[train_inds], y.iloc[test_inds]
    groups_train, groups_test = groups.iloc[train_inds], groups.iloc[test_inds]

    # further split the test set equally: 50% (15%) validation, 50% (15%) test
    val_inds, test_inds = next(GroupShuffleSplit(
        test_size=0.5, random_state=0).split(
            X_test, y_test, groups=groups_test))
    X_val, X_test, y_val, y_test = \
        X_test.iloc[val_inds, :], X_test.iloc[test_inds, :], \
        y_test.iloc[val_inds], y_test.iloc[test_inds]
    groups_val, groups_test = \
        groups_test.iloc[val_inds], groups_test.iloc[test_inds]

    pathout = os.path.join('data', 'features')
    if not os.path.exists(pathout):
        os.makedirs(pathout)
    for Xc, yc, groupsc, ref in zip(
            [X_train, X_val, X_test],
            [y_train, y_val, y_test],
            [groups_train, groups_val, groups_test],
            ['train', 'val', 'test']):
        Xc.to_csv(os.path.join(pathout, f'X_{ref}.csv'), index=False)
        yc.to_csv(os.path.join(pathout, f'y_{ref}.csv'), index=False)
        groupsc.to_csv(os.path.join(pathout, f'groups_{ref}.csv'), index=False)

    return X_train, X_val, X_test, \
        y_train, y_val, y_test, \
        groups_train, groups_val, groups_test


def get_param_grid():
    """Define the grid of parameters to be tested within grid search."""
    param_grid = [
        {
            'scaler': [StandardScaler()],
            'estimator': [LogisticRegression(
                class_weight='balanced', random_state=0,
                fit_intercept=True, solver='lbfgs', max_iter=300)],
            'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        {
            'scaler': [StandardScaler()],
            'estimator': [SGDClassifier(
                class_weight='balanced', loss='log',
                penalty='elasticnet', random_state=0,
                fit_intercept=True, tol=1e-3, max_iter=1000)],
            'estimator__l1_ratio': [0.1, 0.15, 0.2, 0.3],
            'estimator__alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
        },
        {
            'scaler': [StandardScaler()],
            'pca': [PCA(random_state=0)],
            'pca__n_components': [0.8, 0.9, 1],
            'estimator': [SVC(
                probability=True, class_weight='balanced',
                random_state=0, cache_size=20000)],
            'estimator__kernel': ['rbf', 'linear'],
            'estimator__C': [10**x for x in range(-3, 1)]
        },
        {
            'scaler': [DoNothing()],
            'estimator': [DecisionTreeClassifier(
                class_weight='balanced', random_state=0)],
            'estimator__max_features': ['auto', None],
            'estimator__max_depth': [3, 5, 10, 20, 100]
        },
        {
            'scaler': [DoNothing()],
            'estimator': [RandomForestClassifier(
                class_weight='balanced_subsample', random_state=0)],
            'estimator__n_estimators': [50, 100, 500, 1000],
            'estimator__max_depth': [3, 5, 10, 20, 100]},
        {
            'scaler': [DoNothing()],
            'estimator': [LGBMClassifier(
                class_weight='balanced', random_state=0)],
            'estimator__learning_rate': [0.0001, 0.001, 0.01],
            'estimator__colsample_bytree': [0.6, 0.8, 1.0],
            'estimator__n_estimators': [50, 100, 500, 1000],
            'estimator__reg_lambda': [0, 20, 50]
        },
    ]

    return param_grid


def export_best_model(param_grid, X_train, y_train, groups_train):
    """Export best estimator from the grid search cv with param_grid."""
    # initialize pipeline
    pipe = Pipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', DoNothing()),
        ('estimator', LogisticRegression(
            class_weight='balanced', random_state=0))
    ])

    # set up grid search cv with groupkfold
    grd = GridSearchCV(
        pipe, param_grid=param_grid, cv=GroupKFold(), n_jobs=-1,
        scoring='roc_auc', refit=True, return_train_score=True
    )

    # fit grid search with training set
    grd.fit(X_train, y_train, groups_train)

    # export best estimator
    pathout = os.path.join('data', 'model')
    if not os.path.exists(pathout):
        os.makedirs(pathout)
    joblib.dump(
        grd.best_estimator_,
        os.path.join(pathout, 'best_estimator.pkl'),
        compress=1)

    return grd.best_estimator_

if __name__ == '__main__':
    main()

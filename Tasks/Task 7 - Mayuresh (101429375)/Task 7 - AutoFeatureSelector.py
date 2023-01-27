import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2, RFE, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import MinMaxScaler


num_feats = 30


def cor_selector(X, y, num_feats):

    cor_list = []
    feature_name = X.columns.tolist()
    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    cor_support = [True if i in cor_feature else False for i in feature_name]

    return cor_support, cor_feature


def chi_squared_selector(X, y, num_feats):

    X_norm = MinMaxScaler().fit_transform(X)
    chi_selector = SelectKBest(chi2, k=num_feats)
    chi_selector.fit(X_norm, y)
    chi_support = chi_selector.get_support()
    chi_feature = X.loc[:,chi_support].columns.tolist()

    return chi_support, chi_feature


def rfe_selector(X, y, num_feats):

    X_norm = MinMaxScaler().fit_transform(X)
    RFE_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_feats, step=10, verbose=5)
    RFE_selector.fit(X_norm, y)
    rfe_support = RFE_selector.get_support()
    rfe_feature = X.loc[:,rfe_support].columns.tolist()

    return rfe_support, rfe_feature


def embedded_log_reg_selector(X, y, num_feats):

    X_norm = MinMaxScaler().fit_transform(X)
    embedded_lr_selector = SelectFromModel(LogisticRegression(penalty="l1", solver='liblinear'), max_features=num_feats)
    embedded_lr_selector.fit(X_norm, y)
    embedded_lr_support = embedded_lr_selector.get_support()
    embedded_lr_feature = X.loc[:, embedded_lr_support].columns.tolist()

    return embedded_lr_support, embedded_lr_feature


def embedded_rf_selector(X, y, num_feats):

    Xnorm = MinMaxScaler().fit_transform(X)
    embedded_randf_selector = SelectFromModel(RandomForestClassifier(), max_features=num_feats)
    embedded_randf_selector.fit(Xnorm, y)
    embedded_rf_support = embedded_randf_selector.get_support()
    embedded_rf_feature = X.loc[:,embedded_rf_support].columns.tolist()

    return embedded_rf_support, embedded_rf_feature


def embedded_lgbm_selector(X, y, num_feats):

    Xnorm = MinMaxScaler().fit_transform(X)
    lgbc = LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=32, colsample_bytree=0.2, reg_alpha=3, reg_lambda=1, min_split_gain=0.01, min_child_weight=40)
    embd_lgbm_selector = SelectFromModel(lgbc, max_features=num_feats)
    embd_lgbm_selector.fit(Xnorm, y)
    embedded_lgbm_support = embd_lgbm_selector.get_support()
    embedded_lgbm_feature = X.loc[:, embedded_lgbm_support].columns.tolist()

    return embedded_lgbm_support, embedded_lgbm_feature


def preprocess_dataset(dataset_path):

    data = pd.read_csv(dataset_path)
    numcols = ['Overall', 'Crossing', 'Finishing', 'ShortPassing', 'Dribbling', 'LongPassing', 'BallControl',
               'Acceleration', 'SprintSpeed', 'Agility', 'Stamina', 'Volleys', 'FKAccuracy', 'Reactions', 'Balance',
               'ShotPower', 'Strength', 'LongShots', 'Aggression', 'Interceptions']
    catcols = ['Preferred Foot', 'Position', 'Body Type', 'Nationality', 'Weak Foot']
    data = data[numcols + catcols]
    traindf = pd.concat([data[numcols], pd.get_dummies(data[catcols])], axis=1)
    features = traindf.columns
    traindf = traindf.dropna()
    traindf = pd.DataFrame(traindf, columns=features)
    y = traindf['Overall'] > 87
    X = traindf.copy()
    del X['Overall']
    num_feats = 30

    return X, y, num_feats, features


def autoFeatureSelector(dataset_path, methods=[]):
    # Parameters
    # data - dataset to be analyzed (csv file)
    # methods - various feature selection methods we outlined before, use them all here (list)

    # preprocessing
    X, y, num_feats, features = preprocess_dataset(dataset_path)
    feature_name = list(X.columns)

    # Run every method we outlined above from the methods list and collect returned best features from every method
    if 'pearson' in methods:
        cor_support, cor_feature = cor_selector(X, y, num_feats)
    if 'chi-square' in methods:
        chi_support, chi_feature = chi_squared_selector(X, y, num_feats)
    if 'rfe' in methods:
        rfe_support, rfe_feature = rfe_selector(X, y, num_feats)
    if 'log-reg' in methods:
        embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)
    if 'rf' in methods:
        embedded_rf_support, embedded_rf_feature = embedded_rf_selector(X, y, num_feats)
    if 'lgbm' in methods:
        embedded_lgbm_support, embedded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)

    # Combine all the above feature list and count the maximum set of features that got selected by all methods
    feature_selection_df = pd.DataFrame({'Feature': feature_name,
                                         'Pearson': cor_support,
                                         'Chi-2': chi_support,
                                         'RFE': rfe_support,
                                         'Logistics': embedded_lr_support,
                                         'Random Forest': embedded_rf_support,
                                         'LightGBM': embedded_lgbm_support})
    # count the selected times for each feature
    feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
    # display the top 100
    feature_selection_df = feature_selection_df.sort_values(['Total', 'Feature'], ascending=False)
    feature_selection_df.index = range(1, len(feature_selection_df) + 1)
    num_best_feat = 5  # Can be changed according to user preference
    best_features = feature_selection_df["Feature"].tolist()[:num_best_feat]

    return best_features


if __name__ == "__main__":
    best_features = autoFeatureSelector(dataset_path="fifa19.csv",
                                        methods=['pearson', 'chi-square', 'rfe', 'log-reg', 'rf', 'lgbm'])
    print(best_features)
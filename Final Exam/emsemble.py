## import necessary library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

data = pd.read_csv("https://raw.githubusercontent.com/subashgandyer/datasets/main/great_customers.csv")

# Defining dependent(y) and independent(X) variables
X = data.copy()
X = X.drop(columns=['user_id', 'great_customer_class'])
y = data['great_customer_class']


# Label Encoding
## Get the mapping of columns that are going to be encoded
workclass_classes = X['workclass'].unique()
marital_status_classes = X['marital-status'].unique()
occupation_classes = X['occupation'].unique()
race_classes = X['race'].unique()
sex_classes = X['sex'].unique()

le = LabelEncoder()
le.fit_transform(workclass_classes)
workclass_mapping = dict(zip(le.classes_, range(len(le.classes_))))

le.fit_transform(marital_status_classes)
marital_status_mapping = dict(zip(le.classes_, range(len(le.classes_))))

le.fit_transform(occupation_classes)
occupation_mapping = dict(zip(le.classes_, range(len(le.classes_))))

le.fit_transform(race_classes)
race_mapping = dict(zip(le.classes_, range(len(le.classes_))))

le.fit_transform(sex_classes)
sex_mapping = dict(zip(le.classes_, range(len(le.classes_))))

print("Mapping of encoded columns: ")
print(workclass_mapping)
print(marital_status_mapping)
print(occupation_mapping)
print(race_mapping)
print(sex_mapping)

## Label encoder: converting categorical into Numeric
le = LabelEncoder()
Catcols=['workclass', 'marital-status', 'occupation', 'race', 'sex']
X[Catcols] = X[Catcols].apply(le.fit_transform)
X = pd.DataFrame(X)


# Applying Iterative Imputer
imputer = IterativeImputer(max_iter=10, random_state=0)
imputer.fit(X)
X_transform = imputer.transform(X)
X = pd.DataFrame(data=X_transform)

feature_name = list(X.columns)
num_feats = len(feature_name)


# Feature Selection
def cor_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    cor_list = []
    feature_name = X.columns.tolist()

    for i in X.columns.tolist():
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)

    cor_list = [0 if np.isnan(i) else i for i in cor_list]

    cor_feature = X.iloc[:, np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()

    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_feature = cor_selector(X, y,num_feats)

def chi_squared_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    X_norm = MinMaxScaler().fit_transform(X)
    bestfeatures = SelectKBest(score_func=chi2, k=num_feats)
    bestfeatures.fit(X_norm, y)
    chi_support = bestfeatures.get_support()
    chi_feature = X.loc[:, chi_support].columns.tolist()

    # Your code ends here
    return chi_support, chi_feature
chi_support, chi_feature = chi_squared_selector(X, y,num_feats)

def rfe_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    rfe_selector= RFE(estimator=LogisticRegression(),n_features_to_select=num_feats,step=10,verbose=5)
    X_norm=MinMaxScaler().fit_transform(X)
    rfe_selector.fit(X_norm,y)
    rfe_support=rfe_selector.get_support()
    rfe_feature=X.loc[:,rfe_support].columns.tolist()
    # Your code ends here
    return rfe_support, rfe_feature
rfe_support, rfe_feature = rfe_selector(X, y,num_feats)

def embedded_log_reg_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lr_selector=SelectFromModel(LogisticRegression(),max_features=num_feats)
    X_norm=MinMaxScaler().fit_transform(X)
    lr_selector.fit(X_norm,y)
    lr_support=lr_selector.get_support()
    lr_feature=X.loc[:,lr_support].columns.tolist()
    # Your code ends here
    return lr_support, lr_feature
embedded_lr_support, embedded_lr_feature = embedded_log_reg_selector(X, y, num_feats)

def embedded_rf_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100), max_features=num_feats)

    rf_selector.fit(X, y)
    rf_support = rf_selector.get_support()
    rf_feature = X.loc[:, rf_support].columns.tolist()

    # Your code ends here
    return rf_support, rf_feature
embeded_rf_support, embeded_rf_feature = embedded_rf_selector(X, y, num_feats)

def embedded_lgbm_selector(X, y, num_feats):
    # Your code goes here (Multiple lines)
    lgbc=LGBMClassifier(n_estimators=500,learning_rate=.05,num_leaves=32,colsample_bytree=.2,
                       reg_alpha=3,reg_lambda=1,min_split_gain=.01,min_child_weight=40)
    lgb_selector=SelectFromModel(lgbc,max_features=num_feats)
    lgb_selector.fit(X,y)
    lgb_support=lgb_selector.get_support()
    lgb_feature=X.loc[:,lgb_support].columns.tolist()
    # Your code ends here
    return lgb_support, lgb_feature
embeded_lgbm_support, embeded_lgbm_feature = embedded_lgbm_selector(X, y, num_feats)

# Feature Selection Summary
feature_selection_df = pd.DataFrame({'Feature':feature_name, 'Pearson':cor_support, 'Chi-2':chi_support, 'RFE':rfe_support, 'Logistics':embedded_lr_support,
                                    'Random Forest':embeded_rf_support, 'LightGBM':embeded_lgbm_support})
# count the selected times for each feature
feature_selection_df['Total'] = np.sum(feature_selection_df, axis=1)
# display the top 100
feature_selection_df = feature_selection_df.sort_values(['Total','Feature'] , ascending=False)
feature_selection_df.index = range(1, len(feature_selection_df)+1)
feature_selection_df.head(num_feats)

X = X.drop(columns=[0, 1, 2, 3, 4])


# Normalising and Splitting the dataset into Train and Test
X_norm=MinMaxScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.20, random_state=0, shuffle=True)

#Defining and Training the Random Forest Classifiesr
rf_model = RandomForestClassifier(n_estimators=100,
                                  random_state=0,
                                  max_features='sqrt',
                                  n_jobs=-1,
                                  verbose=1)
rf_model.fit(X_train, y_train)

# Node count and Maximum depth of the tree
n_nodes = []
max_depths = []

for ind_tree in rf_model.estimators_:
    n_nodes.append(ind_tree.tree_.node_count)
    max_depths.append(ind_tree.tree_.max_depth)

print(f'Average number of nodes {int(np.mean(n_nodes))}')
print(f'Average maximum depth {int(np.mean(max_depths))}')

#Make Random Forest predictions
train_rf_predictions = rf_model.predict(X_train)
train_rf_probs = rf_model.predict_proba(X_train)[:, 1]

rf_predictions = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

#Plot ROC AUC Score
print(f'Train ROC AUC Score: {roc_auc_score(y_train, train_rf_probs)}')
print(f'Test ROC AUC  Score: {roc_auc_score(y_test, rf_probs)}')
print(f'Baseline ROC AUC: {roc_auc_score(y_test, [1 for _ in range(len(y_test))])}')

# Calculate the accuracy score
y_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {rf_accuracy}")

#Random Forest Confusion Matrix
rf_cm = confusion_matrix(y_test, rf_predictions)
print("Confusion Matrix:\n", rf_cm)

# Defining and Training the Support Vector Machine Model
svm_model = SVC(random_state=0)
svm_model.fit(X_train, y_train)

# Testing the Stochastic Gradient Descent Model
svm_predictions = svm_model.predict(X_test)
svm_score = svm_model.score(X_test, y_test)
print("Accuracy: ", svm_score)

svm_cm = confusion_matrix(y_test,svm_predictions)
print("Confusion Matrix:\n", svm_cm)

# Defining and Training the logistic Regression Model
lr_model = LogisticRegression(random_state=0)
lr_model.fit(X_train, y_train)

# Testing the Logistic Regression Model
lr_predictions = lr_model.predict(X_test)
lr_score = lr_model.score(X_test, y_test)
print("Accuracy: ", lr_score)

# Confusion Matrix
lr_cm = confusion_matrix(y_test,lr_predictions)
print("Confusion Matrix:\n", lr_cm)


nb_model = GaussianNB()
nb_model.fit(X_train, y_train)

# Testing the Logistic Regression Model
nb_predictions = nb_model.predict(X_test)
nb_score = nb_model.score(X_test, y_test)
print("Accuracy: ", nb_score)

# Confusion Matrix
nb_cm = confusion_matrix(y_test,nb_predictions)
print("Confusion Matrix:\n", nb_cm)


knn_model = KNeighborsClassifier(n_neighbors=5, metric="minkowski",p=2)
knn_model.fit(X_train, y_train)

# Testing the Logistic Regression Model
knn_predictions = knn_model.predict(X_test)
knn_score = knn_model.score(X_test, y_test)
print("Accuracy: ", knn_score)

# Confusion Matrix
knn_cm = confusion_matrix(y_test,knn_predictions)
print("Confusion Matrix:\n", knn_cm)


# # Best Evalutation Metric:
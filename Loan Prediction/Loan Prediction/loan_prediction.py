import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.learning_curve import learning_curve
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
# Data Pre-processing
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

all_data = pd.concat([train, test], keys=['train', 'test'])


# train[train.dtypes[(train.dtypes == "float64") | (train.dtypes == "int64")].index.values].hist(figsize=[11, 11])


def clean_data_set(data_set):
    data_set['ApplicantIncome'].fillna(data_set['ApplicantIncome'].mean(), inplace=True)
    data_set['CoapplicantIncome'].fillna(data_set['CoapplicantIncome'].mean(), inplace=True)
    data_set['LoanAmount'].fillna(data_set['LoanAmount'].mean(), inplace=True)
    data_set['Loan_Amount_Term'].fillna(360, inplace=True)
    data_set['Gender'].fillna('Male', inplace=True)
    data_set['Married'].fillna('No', inplace=True)
    data_set['Dependents'].fillna(0, inplace=True)
    data_set['Self_Employed'].fillna('No', inplace=True)
    data_set['not_self_graduate_married'] = data_set.apply(lambda x: 1 if (x['Self_Employed'] == 'No') & (x['Married'] == 'Yes') & (x['Education'] == 'Graduate') & (x['Dependents'] <=1) else 0, axis=1)
    data_set['Dependents'] = data_set['Dependents'].apply(lambda x: 3 if x == '3+' else x)
    data_set['Credit_History'].fillna(data_set['Credit_History'].mean(), inplace=True)
    data_set['sum_income'] = data_set['ApplicantIncome'] + data_set['CoapplicantIncome']
    data_set['EMI'] = (data_set['LoanAmount'] * 0.095 * ((1+0.095) ** data_set['Loan_Amount_Term'])) \
                      /(1.095 ** data_set['Loan_Amount_Term'] -1)
    data_set['Ratio'] = data_set['LoanAmount']/data_set['sum_income']
    data_set['R_6_Unmarried'] = data_set.apply(lambda x: 1 if (x['Ratio'] > 6) & (x['Married'] == 'No') else 0, axis=1)
    data_set['male_6000_not_graduate'] = data_set.apply(lambda x: 1 if (x['ApplicantIncome'] < 6000) & (x['Gender'] == 'Male') & (x['Education'] == 'Not Graduate') else 0, axis=1)
    data_set['female_self'] = data_set.apply(lambda x: 1 if (x['Gender'] == 'Female') & (x['Self_Employed'] == 'Yes') else 0, axis=1)

    return data_set

# One-hot Encoding
clean_all = clean_data_set(all_data)


def one_hot_encoding(data_set):
    le = LabelEncoder()
    var_to_encode = ['Gender','Married','Education','Self_Employed','Property_Area']
    for col in var_to_encode:
        data_set[col] = le.fit_transform(data_set[col])
    
    data_set = pd.get_dummies(data_set, columns=var_to_encode)
    return data_set

# Data Split into training and testing
clean_all = one_hot_encoding(clean_all)

clean_train = clean_all.ix['train']
clean_train['Loan_Status'] = clean_train['Loan_Status'].apply(lambda x: 1 if x == 'Y' else 0)
clean_test = clean_all.ix['test']
clean_test = clean_test.ix[:, clean_test.columns != 'Loan_Status']


def split_data(data_set, response_col):
    x, y = data_set.iloc[:, ~data_set.columns.isin([response_col, 'Loan_ID'])], data_set[response_col]
    p_train, p_test, v_train, v_test = train_test_split(x, y, test_size=0.30, random_state=0)
    return p_train, p_test, v_train, v_test

x_train, x_test, y_train, y_test = split_data(clean_train, 'Loan_Status')

# Random Forest Algorithm

# clf = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=4, criterion='entropy')
# clf.fit(x_train, y_train)
# print 'Test Accuracy: %.3f' % clf.score(x_test, y_test)

# GBM model

gbm = sklearn.ensemble.GradientBoostingClassifier(n_estimators=10, max_depth=3)
gbm.fit(x_train, y_train)
print 'Test Accuracy: %.3f' % gbm.score(x_test, y_test)

# # Learning Curve
# train_sizes, train_scores, test_scores =\
#     learning_curve(estimator=clf,
#                    X=x_train,
#                    y=y_train,
#                    train_sizes=np.linspace(0.1, 1.0, 10),
#                    cv=10,
#                    n_jobs=1)
#
# train_mean = np.mean(train_scores, axis=1)
# train_std = np.std(train_scores, axis=1)
# test_mean = np.mean(test_scores, axis=1)
# test_std = np.std(test_scores, axis=1)
# plt.plot(train_sizes, train_mean, color = 'blue', marker='o', markersize=5,
#          label='training accuracy')
# plt.fill_between(train_sizes, train_mean + train_std,
#                  train_mean - train_std, alpha=0.15, color='blue')
# plt.plot(train_sizes, test_mean, color='green', linestyle='--',
#          marker='s', markersize=5, label='validation accuracy')
# plt.fill_between(train_sizes, test_mean + test_std,
#                  test_mean - test_std,
#                  alpha=0.15, color='green')
# plt.grid()
# plt.xlabel('Number of training samples')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')
# plt.ylim([0.8, 1.0])
# plt.show()

# Performance Report

def show_performance(x_train, x_test, y_train, y_test, model):
    # model.fit(training_set[predictors], training_set['Survived'])
    predictions = model.predict(x_train)
    probabilities = model.predict_proba(x_train)[:, 1]
    print 'Performance Report'
    print 'Accuracy : {:.2%}'.format(metrics.accuracy_score(y_train, predictions))
    print 'AUC Score (train): {0}'.format(metrics.roc_auc_score(y_train, probabilities))

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]
    print 'Performance Report'
    print 'Accuracy : {:.2%}'.format(metrics.accuracy_score(y_test, predictions))
    print 'AUC Score (Test): {0}'.format(metrics.roc_auc_score(y_test, probabilities))


show_performance(x_train, x_test, y_train, y_test, gbm)

# Confusion Matrix
from sklearn.metrics import confusion_matrix
gbm.fit(x_train, y_train)
y_pred = gbm.predict(x_test)
confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
print confmat

# Build a forest and compute the feature importances
#
# gbm.fit(x_train, y_train)
# importances = gbm.feature_importances_
# std = np.std([tree.feature_importances_ for tree in gbm.estimators_],
#              axis=0)
# indices = np.argsort(importances)[::-1]
#
# # Print the feature ranking
# print("Feature ranking:")
#
# for f in range(x_train.shape[1]):
#     print("%d. feature %s (%f)" % (f + 1, x_train.columns[indices[f]], importances[indices[f]]))
#
# # Plot the feature importances of the gbm
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(x_train.shape[1]), importances[indices],
#         color="r", yerr=std[indices], align="center")
# plt.xticks(range(x_train.shape[1]), [x_train.columns[x] for x in indices])
# plt.xlim([-1, x_train.shape[1]-9])
# plt.show()


# Cleaning and Predicting the testing set
clean_test['Loan_Status'] = gbm.predict(clean_test.loc[:, clean_test.columns != 'Loan_ID'])
clean_test['Loan_Status'] = clean_test['Loan_Status'].apply(lambda x: 'Y' if x == 1 else 'N')
clean_test.to_csv('result.csv')








# Predict Credit History
train = clean_data_set(train)
train = train.ix[:, train.columns != 'Loan_Status']
train_pre, train_test = train[~train['Credit_History'].isnull()], train[train['Credit_History'].isnull()]

train_pre = one_hot_encoding(train_pre)


train_pre['Credit_History'] = train_pre['Credit_History'].astype(int)

c_train, c_test, cv_train, cv_test = split_data(train_pre, 'Credit_History')



his = sklearn.ensemble.RandomForestClassifier(n_estimators=200, max_depth=25)
his.fit(c_train, cv_train)
print 'Test Accuracy: %.3f' % his.score(c_test, cv_test)


def show_performance1(x_train, x_test, y_train, y_test, model, thred):
    # model.fit(training_set[predictors], training_set['Survived'])
    predictions = (model.predict_proba(x_train)[:,1] > thred).astype(int)
    probabilities = model.predict_proba(x_train)[:, 1]
    print 'Performance Report'
    print 'Accuracy : {:.2%}'.format(metrics.accuracy_score(y_train, predictions))
    print 'AUC Score (train): {0}'.format(metrics.roc_auc_score(y_train, probabilities))

    predictions = (model.predict_proba(x_test)[:,1] > thred).astype(int)
    probabilities = model.predict_proba(x_test)[:, 1]
    print 'Performance Report'
    print 'Accuracy : {:.2%}'.format(metrics.accuracy_score(y_test, predictions))
    print 'AUC Score (Test): {0}'.format(metrics.roc_auc_score(y_test, probabilities))
    y_pred = (his.predict_proba(x_test)[:, 1] > thred).astype(int)
    confmat = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print confmat

show_performance1(c_train, c_test, cv_train, cv_test, his, 0.5)






train_test = train_test.ix[:, train_test.columns != 'Credit_History']

train_test = one_hot_encoding(train_test)
train_test['Credit_History'] = his.predict(train_test)

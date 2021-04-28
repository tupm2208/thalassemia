import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor



# numerical_columns = ["SLHC","HST","HCT","MCV","MCH","MCHC","RDWCV","SLTC","SLBC"]
# numerical_columns = ["FERRITIN","FE"]
# numerical_columns = ["SLHC","HST","HCT","MCV","MCH","MCHC","RDWCV","SLTC","SLBC","FERRITIN","FE"]
numerical_columns = ["FERRITIN","FE", "HBA1", "HBA2"]

def read_main_data(file_path = 'datasets/processed_data.csv'):
    global numerical_columns

    df = pd.read_csv(file_path, low_memory=False)
    df.set_index("PID", inplace=True)
    
    num_df = df[numerical_columns].apply(pd.to_numeric, errors='coerce', downcast='float')
    df = df.reindex(num_df.dropna().index)

    df[numerical_columns] = num_df.dropna()
    
    train_df, test_df = train_test_split(df[numerical_columns + ['thalas']], test_size=0.2, stratify=df.thalas, random_state=42)

    return train_df, test_df


def read_additional_data(file_path):
    global numerical_columns

    df = pd.read_csv(file_path, low_memory=False)
    # df.drop_duplicates(subset='PID')
    # df.set_index("PID", inplace=True)

    num_df = df[numerical_columns].apply(pd.to_numeric, errors='coerce', downcast='float')
    feature_df = num_df.dropna()
    label_df = df.loc[feature_df.index].reset_index(drop=True)
    feature_df.reset_index(drop=True, inplace=True)

    label = (label_df['alpha'] == 1) | (label_df['beta'] == 1)
    label = label.astype(int)
    
    out_df = pd.concat([feature_df, pd.DataFrame({'thalas': label.tolist()})], axis=1)
    return out_df

def balance_data(df, type='over'):
    positive_df = df[df.thalas==1].reset_index(drop=True)
    negative_df = df[df.thalas==0].reset_index(drop=True)

    p_size = positive_df.shape[0]
    n_size = negative_df.shape[0]
    if p_size > n_size:
        s = np.random.randint(n_size, size=p_size)
        negative_df = negative_df.loc[s].reset_index(drop=True)
    elif p_size < n_size:
        s = np.random.randint(p_size, size=n_size)
        positive_df = positive_df.loc[s].reset_index(drop=True)
    
    return pd.concat([positive_df, negative_df]).reset_index(drop=True)

def fit_model(model, df):
    global numerical_columns

    df = balance_data(df)
    X = df[numerical_columns]
    y = df['thalas']

    # model.partial_fit(X, y)
    model.fit(X, y)


def evaluate(model, df, desc=''):
    global numerical_columns
    
    X = df[numerical_columns]
    y = df['thalas']
    
    preds = np.array(model.predict(X))
    print('---------------------------------------------------')
    print(f"evaluation results {desc}:\n")
    print(classification_report(y, np.where(preds>=0.5, 1, 0)))
    print('---------------------------------------------------')

    # mask = preds != y
    mask = (0.3 <= preds) & (preds <= 0.7)
    print(sum(mask))

    features_df = df[mask][numerical_columns].reset_index(drop=True)
    labels = preds[mask]
    labels = np.where(labels>=0.5, 1, 0)
    # print(labels)
    new_df = pd.concat([features_df, pd.DataFrame({'thalas': labels})], axis=1)

    return new_df, df[preds == y].reset_index(drop=True)


def main():
    train_df, test_df = read_main_data()

    # model = xgb.XGBClassifier(learning_rate=0.1, max_depth=10, n_estimators=100, n_jobs=-1, random_state=42,use_label_encoder=False, objective="binary:logistic")
    # model = RandomForestClassifier(n_estimators = 100, random_state=42, criterion='entropy', n_jobs=8)
    # model = GaussianNB()
    # model = RandomForestRegressor(n_estimators = 100, random_state=42,n_jobs=8)
    # weighted = train_df.shape[0] / (2 * np.bincount(train_df.thalas))
    # model = svm.SVC(C=0.5,kernel='rbf', class_weight={0: weighted[0], 1: weighted[1]})
    # model = svm.SVC(C=0.5,kernel='rbf')
    xg = xgb.XGBClassifier(learning_rate=0.1, max_depth=4, n_estimators=100, n_jobs=-1, random_state=42,use_label_encoder=False, objective="binary:logistic")
    rf = RandomForestClassifier(n_estimators = 100, random_state=42, criterion='entropy', n_jobs=8)
    nb = GaussianNB()
    model = VotingClassifier(estimators=[('rf', rf), ('nb', nb)], voting='soft')

    
    df2 = read_additional_data('datasets/0504.csv')
    df3 = read_additional_data('datasets/1904.csv')

    # test_df = pd.concat([test_df, df3])
    df2 = pd.concat([df2, df3])
    fit_model(model, train_df)
    evaluate(model, test_df, 'main_test')
    outlier_df, df2 = evaluate(model, df2, 'outlier')
    fit_model(model, pd.concat([train_df, outlier_df]))
    evaluate(model, test_df, "MAIN_TEST")
    evaluate(model, df2, 'outlier')

if __name__ == '__main__':
    # read_additional_data('datasets/0504.csv')
    main()
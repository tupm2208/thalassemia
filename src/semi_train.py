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
from sklearn.neighbors import KNeighborsClassifier



numerical_columns = ["SLHC","HST","HCT","MCV","MCH","MCHC","RDWCV","SLTC","SLBC"]
# numerical_columns = ["FERRITIN","FE"]
numerical_columns = ["SLHC","HST","HCT","MCV","MCH","MCHC","RDWCV","SLTC","SLBC","FERRITIN","FE"]
# numerical_columns = ["SLHC","HST","HCT","MCV","MCH","MCHC","RDWCV","SLTC","SLBC","FERRITIN","FE", "HBA1", "HBA2"]
# numerical_columns = ["FERRITIN","FE", "HBA1", "HBA2"]
# categorical_columns = ['sex']
categorical_columns = []

def read_main_data(file_path = 'datasets/splited_train_test.csv'):
    global numerical_columns

    df = pd.read_csv(file_path, low_memory=False)
    df.drop_duplicates(subset=['PID'], keep='last', inplace=True)
    # df = df[~(df.sheet.isin(['d1']) & (df.thalas == 0)& (df.train == 1))]
    # df1 = df[~df.sheet2.isin(['normal'])]
    # df2 = df[df.sheet2.isin(['normal'])]
    # df = pd.concat([df1, df2.iloc[:2000]])
    # df = pd.concat([df1, df2])
    # df = df1
    # print(df1.shape, df2.shape)

    # df = df[((df.MCV<85) | (df.MCH<28))]
    # df.set_index("PID", inplace=True)
    
    num_df = df[numerical_columns].apply(pd.to_numeric, errors='coerce', downcast='float')
    # num_df = df[["SLHC","HST","HCT","MCV","MCH","MCHC","RDWCV","SLTC","SLBC","FERRITIN","FE"]].apply(pd.to_numeric, errors='coerce', downcast='float')
    
    df = df.reindex(num_df.dropna().index)

    # df[["SLHC","HST","HCT","MCV","MCH","MCHC","RDWCV","SLTC","SLBC","FERRITIN","FE"]] = num_df.dropna()
    df[numerical_columns] = num_df.dropna()

    # train_df, test_df = train_test_split(df[numerical_columns + ['thalas']], test_size=0.2, stratify=df.thalas, random_state=42)

    test_df = df[df.train==0]
    train_df = df[df.train==1]
    # train_df = train_df[~train_df.PID.isin(test_df.PID)]
    # train_df = train_df[~train_df.PID.isin(test_df.PID) & ((train_df.MCV<85) | (train_df.MCH<28))]
    # print(train_df['MCV'].describe())
    

    return train_df, test_df

def process_categorical(df, num_df):
    global categorical_columns
    arr_df = [pd.get_dummies(df[e]) for e in categorical_columns]
    return pd.concat([num_df] + arr_df, axis=1)

def process_features(df):
    global categorical_columns, numerical_columns
    X = df[numerical_columns]
    X = process_categorical(df, X)
    # X = X.assign(type=np.where((X.MCV>=85) & (X.MCH>=28), 1, 0))
    return X


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

def balance_data(df, type='under'):
    positive_df = df[df.thalas==1].reset_index(drop=True)
    negative_df = df[df.thalas==0].reset_index(drop=True)

    p_size = positive_df.shape[0]
    n_size = negative_df.shape[0]

    
    if type == 'under':
        sz = min(p_size, n_size)
    elif type == 'over':
        sz = max(p_size, n_size)
    else:
        sz = type
    if n_size != sz:
        s = np.random.randint(n_size, size=sz)
        negative_df = negative_df.loc[s].reset_index(drop=True)
    if p_size != sz:
        s = np.random.randint(p_size, size=sz)
        positive_df = positive_df.loc[s].reset_index(drop=True)
    
    return pd.concat([positive_df, negative_df]).reset_index(drop=True)


def evaluate(model, df, desc=''):
    global numerical_columns
    
    X = process_features(df)
    y = df['thalas']
    
    preds = np.array(model.predict(X))
    print('---------------------------------------------------')
    print(f"evaluation results {desc}:\n")
    print(classification_report(y, np.where(preds>=0.6, 1, 0)))
    print('---------------------------------------------------')

    # mask = preds != y
    mask = (0.3 <= preds) & (preds <= 0.7)
    print(sum(mask))

    features_df = df[mask][numerical_columns].reset_index(drop=True)
    labels = preds[mask]
    labels = np.where(labels>=0.8, 1, 0)
    # print(labels)
    new_df = pd.concat([features_df, pd.DataFrame({'thalas': labels})], axis=1)

    return new_df, df[preds == y].reset_index(drop=True)


def fit_model(model, df):
    global numerical_columns

    # df = balance_data(df, 5000)
    # df = balance_data(df, 'under')
    # df = balance_data(df, 'over')
    X = process_features(df)
    y = df['thalas']

    model.fit(X, y)

    print('positive: ', (y==1).sum())
    print('negative: ', (y==0).sum())


def main():
    train_df, test_df = read_main_data()

    # model = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, n_estimators=100, n_jobs=-1, random_state=42,use_label_encoder=False, objective="binary:logistic")
    model = xgb.XGBRegressor(learning_rate=0.1, max_depth=15, n_estimators=100, n_jobs=-1, random_state=42,use_label_encoder=False, objective="binary:logistic")
    # model = RandomForestClassifier(n_estimators = 100, random_state=42, criterion='entropy', n_jobs=8)
    # model = GaussianNB()
    # model = RandomForestRegressor(n_estimators = 100, random_state=42,n_jobs=8)
    # weighted = train_df.shape[0] / (2 * np.bincount(train_df.thalas))
    # model = svm.SVC(C=0.1,kernel='rbf', class_weight={0: weighted[0], 1: weighted[1]})
    # model = svm.SVC(C=10,kernel='rbf', class_weight={0: weighted[0], 1: weighted[1]})
    # model = svm.SVC(C=200,kernel='rbf')
    # model = KNeighborsClassifier(n_neighbors=4)

    # svm_model = svm.SVC(C=200,kernel='rbf', probability=True)
    # xg = xgb.XGBClassifier(learning_rate=0.1, max_depth=15, n_estimators=100, n_jobs=-1, random_state=42)
    # rf = RandomForestClassifier(n_estimators = 100, random_state=42, criterion='entropy', n_jobs=8)
    # nb = GaussianNB()
    # model = VotingClassifier(estimators=[('rf', rf), ('nb', nb), ('xg', xg)], voting='soft')
    # model = VotingClassifier(estimators=[('rf', rf), ('xg', xg)], voting='soft')

    
    # test_df = pd.concat([test_df, df3])
    # df2 = pd.concat([df2, df3])
    fit_model(model, train_df)
    evaluate(model, test_df, 'main_test')
    evaluate(model, train_df, 'main_test')
    import pickle
    with open('models/model_fe.pkl', 'wb') as f:
        pickle.dump(model, f)

if __name__ == '__main__':
    # read_additional_data('datasets/0504.csv')
    main()
import enum
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from glob import glob
from mapping import feature_names, alpha_names, beta_names, mp, pattern, basic_names, fe_names


def handle_feature(df, f_names, obj):
    intersection_names = set(f_names) & set(df.columns)
    feature_df = df[intersection_names]
    feature_df = feature_df.apply(pd.to_numeric, errors='coerce', downcast='float')
    sorted_df = feature_df.loc[feature_df.isnull().sum(axis=1).sort_values().index]
    values = sorted_df.iloc[0].to_dict()
    obj.update(values)

def handle_label(df, l_names, obj):
    new_df = df
    intersection_columns = set(l_names) & set(new_df.columns)
    transpose_df = new_df[intersection_columns]
    transpose_df = transpose_df.replace(regex='DƯƠNG', value=-2)
    transpose_df = transpose_df.replace(regex='ÂM', value=-1)
    
    vl = transpose_df.replace(regex='x', value=np.nan).fillna(method='bfill').iloc[0].to_dict()
    obj.update(vl)

    return len(intersection_columns) != 0

def process_header(df):
    target_lb = list(pattern.keys())

    for idx, e in enumerate(df.columns.values):
        for k in target_lb:
            if k.lower() in str(e).lower():
                df.columns.values[idx] = k
                break

def process_input(file_path):
    df = pd.read_excel(file_path)
    if 'Unnamed: 0' == df.columns[0] and 'x' in df.columns:
        df.drop(columns=['Unnamed: 0'], inplace=True)
        df.columns.values[0] = 'Mã XN'
    if "x" in df.columns:
        df.columns.values[0] = 'Mã XN'
    
    df.columns = df.columns.str
    if 'Tên xét nghiệm' in df.columns:
        df.drop(columns=['Tên xét nghiệm'], inplace=True)
    # print(df.head())
    # print(df.columns)
    if 'Mã XN' not in df.columns:
        return None
    print(df.columns)
    df = df.set_index(df.columns.values[0]).transpose()
    df = df.iloc[::-1]
    
    if df.shape[0] == 0:
        return None
    
    if '233RDW-CV' not in set(df.columns.values):
        for idx, e in enumerate(df.columns.values):
            if str(e) == '236':
                df.columns.values[idx] = '233RDW-CV'
                break
    
    process_header(df)
    
    return df


def handle_data(file_path):
    df = process_input(file_path)
    
    if df is None:
        return None
    obj = pattern.copy()
    obj['PID'] = os.path.basename(file_path).split('.')[0]
    
    handle_feature(df, basic_names, obj)
    handle_feature(df, fe_names, obj)
    handle_feature(df, feature_names, obj)
    have_beta = handle_label(df, beta_names, obj)
    have_alpha = handle_label(df, alpha_names, obj)

    # if not have_alpha or not have_beta:
    #     return None

    return obj.values()


def post_process_data(df, out_file_path='out.csv'):
    f_names = basic_names + fe_names + feature_names
    feature_df = df[f_names]
    feature_df.columns = [mp[e] for e in f_names]
    alpha_df = df[alpha_names]
    beta_df = df[beta_names]

    beta_arr = []
    alpha_arr = []

    for idx in range(df.shape[0]):
        alpha_arr.append(1 if -2 in alpha_df.iloc[idx].values else 0)
        beta_arr.append(1 if -2 in beta_df.iloc[idx].values else 0)

    label_df = pd.DataFrame({
        "alpha": alpha_arr,
        'beta': beta_arr
    })

    out_df = pd.concat([feature_df, label_df], axis=1)
    return out_df.dropna(axis=0, thresh=int(0.6*out_df.shape[1]), how='all')


def main(gl_path, csv_path):
    file_paths = glob(gl_path)
    main_arr = []

    for file_path in tqdm(file_paths):
        try:
            out = handle_data(file_path)
            if out is None:
                print(file_path)
                continue
            main_arr.append(out)
        except Exception as e:
            print(f'error: {e}', file_path)
    df = pd.DataFrame(main_arr, columns=pattern.keys())
    df = post_process_data(df)
    df.to_csv(csv_path, index=False)
    

if __name__ == "__main__":
    names = '2804'
    file_path = f"/home/tupm/datasets/thalas/{names}/*/13001139*"
    # main(file_path, f'../datasets/{names}.csv')
    main(file_path, f'/home/tupm/projects/thalassemia/datasets/test.csv')
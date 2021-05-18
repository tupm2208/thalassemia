import pandas as pd
from glob import glob

path = 'datasets/OneDrive_1_5-10-2021/1. XN_CBC_Tổng phân tích tế bào máu/*modify.csv'

columns = ['intime', 'PID', 'SID', 'Age', 'sex', 'Address', 'LocationID', 'LocationName', 'Testcode', 'Testname', 'Result']

file_paths = glob(path)

def get_df(file_path):
    print(file_path)
    df = pd.read_csv(file_path, names=columns, low_memory=False)
    
    df.PID = df.PID.apply(pd.to_numeric,errors='coerce', downcast='integer')
    df.PID = df.PID.astype(str).str.split('.').str[0]
    df.drop_duplicates(subset='PID', inplace=True)
    # df.set_index('', inplace=True)
    df = df[['PID', 'Age', 'sex']]
    return df


df_list = [get_df(e) for e in file_paths]
df = pd.concat(df_list)
df.drop_duplicates(subset='PID', inplace=True)
df.to_csv('datasets/address.csv', index=False)
print(df.shape)
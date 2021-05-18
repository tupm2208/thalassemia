from os import name
import pandas as pd
from tqdm import tqdm

path = "datasets/pid.csv"
df = pd.read_csv(path)
flag_dict = {e.strip(): True for e in df['PID']}
columns = ['intime', 'PID', 'SID', 'Age', 'sex', 'Address', 'LocationID', 'LocationName', 'Testcode', 'Testname', 'Result']
pattern = None

def process_line(source_path, des_path):
    f2 = open(des_path, 'w+', encoding='utf8')
    with open(source_path, 'r', encoding='utf8') as f:
        lines = f.readlines()
        
        out = []
        for line in tqdm(lines):
            splited = line.split(',')
            out = splited[:5] + ['-'.join(splited[5:-5])] + splited[-5:]
            PID = out[1].strip()
            out[1] = PID
            if PID is not None and flag_dict.get(PID):
                f2.write(','.join(out))
    f2.close()
    print("file converted!!!")


def process_csv(source_path):
    global pattern
    with open(source_path, 'r', encoding='utf8') as f:
        lines = f.readlines()

        arr = []
        for line in tqdm(lines):
            line = line.strip()
            out = line.split(',')
            PID = out[1].strip()
            out[1] = PID
            if PID is not None and flag_dict.get(PID):
                arr.append(out)
    
    df = pd.DataFrame(arr, columns=columns)
    # df.set_index('PID', inplace=True)
    df = df.assign(Testcode=df.Testcode.str.strip())
    pattern = {e: None for e in df.Testcode.unique()}
    return df

def handle_group(ex_df):
    global pattern
    obj = pattern.copy()
    df2 = ex_df[['intime', 'Testcode', 'Result']]
    df2 = df2.assign(indate=df2['intime'].str.split().str[0])
    df2.apply(pd.to_numeric, errors='coerce', downcast='float')
    # df2.dropna(inplace=True)
    m = 0
    selected_df = None
    for date, sub_df in df2.groupby('indate'):
        if m <= sub_df.shape[0]:
            m = sub_df.shape[0]
            obj['indate'] = date
            selected_df = sub_df

    main_df = selected_df[['Testcode','Result']]
    main_dict = main_df.set_index('Testcode').to_dict()['Result']
    obj.update(main_dict)
    return obj

def process_dataframe(df, des_path):
    out_arr = []
    for pid, sub_df in tqdm(df.groupby("PID")):
        pid = str(int(pid)).strip()
        if pid is not None and flag_dict.get(pid):
            tgt = handle_group(sub_df)
            tgt['PID'] = pid
            out_arr.append(tgt)
    
    tgt_df = pd.DataFrame(out_arr).set_index("PID")
    tgt_df.to_csv(des_path)

def handle_xlsx(file_path, des_folder):
    global pattern
    # xl = pd.ExcelFile(file_path)
    # for e in tqdm(xl.sheet_names):
    for e in ['SH-2013', 'SH-2015', 'SH-2016', 'SH-2017', 'SH-2018', 'SH-2019', 'SH-2020']:
        des_path =  f'{des_folder}/{e}.csv'
        csv_path = f'{os.path.dirname(file_path)}/{e}.csv'
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
        else:
            df = xl.parse(e)
            df.to_csv(csv_path, index=False)
            pass
        pattern = {e: None for e in df.Testcode.unique()}
        process_dataframe(df, des_path)

def main(source_path, des_path):
    if source_path.endswith('.csv'):
        df = process_csv(source_path)
        process_dataframe(df, des_path)
    elif source_path.endswith('.xlsx'):
        handle_xlsx(source_path, des_path)



if __name__ == '__main__':
    from glob import glob
    import os

    file_paths = glob('datasets/OneDrive_1_5-10-2021/4. XN_PCR_Di truyền sinh học phân tử/*')
    for path in file_paths:
        if 'modify' in path:
            continue
        print(path)
        inte_path = path.replace(".csv", '_modify.csv')
        des_path = f'datasets/dien_di/{os.path.basename(path)}'
        # process_line(path, inte_path)
        main(inte_path, des_path)

    # path = 'datasets/OneDrive_1_5-10-2021/2. XN_Sinh hóa máu/SINH HOA-VHH2013-2019.xlsx'
    # des_path = 'datasets/SINH_HOA'
    # main(path, des_path)

    # path = 'datasets/OneDrive_1_5-10-2021/1. XN_CBC_Tổng phân tích tế bào máu/TB.2015.csv'
    # inte_path = path.replace(".csv", '_modify.csv')
    # des_path = 'datasets/TBM/TB.2015.csv'
    # process_line(path, inte_path)
    # main(inte_path, des_path)
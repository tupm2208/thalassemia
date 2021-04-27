import pandas as pd
from glob import glob
import os
from tqdm import tqdm
from multiprocessing import Pool

file_paths = glob('/home/tupm/datasets/thalas/2704/*/*')

def action(file_path):
    out_path = file_path.replace('2704', '2704_copy')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = pd.read_excel(file_path)
    df.to_excel(out_path)

p = Pool(8)

for _ in tqdm(p.imap(action, file_paths)):
    pass

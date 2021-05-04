import numpy as np
import pandas as pd

from mapping import mp

original_data_file = 'datasets/pid.xlsx'
xl = pd.ExcelFile(original_data_file)
print(xl.sheet_names)
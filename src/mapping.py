feature_names = ['750HbA1', '751HbA2', '752HbE', '753HbF']

alpha_names = ['SEA', 'A32431', '3.7', 'A32428', '4.2', 'A32426', 'THAI', 'A32432', 'FIL', 'HbCS', 'A32427', 'HbQs', 'A32441', '2delT', 'A32429']

beta_names = ['-28', 'cd 27/28', 'CD28', 'cd15', 'cd 17', 'CD17', 'A32435', 'Co17', 'cd 19', 'cd 8/9', 'cd 26', 'CD26', 'Co26', 'A32433', '41/42', 'A32442', 'cd 43', '71/72', 'cd 95', 'CD95', 'cd121', 'IVS', 'A32440', '89/90', 'cd90', '-88', 'b-29', 'b-31']

basic_names = ['PID', '223SLHC', '224HST', '225HCT', '226MCV', '227MCH', '228MCHC', '233RDW-CV', '234SLTC', '246SLBC']
# basic_names = ['PID', '223SLHC', '224HST', '225HCT', '226MCV', '227MCH', '228MCHC', '236', '234SLTC', '246SLBC']
fe_names = ['1012FE', '1075SAT']



mp = {
    "PID": "PID",
    "223SLHC": "SLHC",
    "224HST": "HST",
    "225HCT": "HCT",
    "226MCV": "MCV",
    "227MCH": "MCH",
    "228MCHC": "MCHC",
    "233RDW-CV": "RDWCV",
    # "236": "RDWCV",
    "234SLTC": "SLTC",
    "246SLBC": "SLBC",
    "1075SAT": "FE",
    "1012FE": "FERRITIN",
    "750HbA1": "HBA1",
    "751HbA2": "HBA2",
    "752HbE": "HBE",
    "753HbF": "HBF",

 

    "SEA": "SEA",
    "A32431": "SEA",
    "3.7": "A37",
    "A32428": "A37",
    "4.2": "A42",
    "A32426": "A42",
    "THAI": "THAI",
    "A32432": "THAI",
    "FIL": "FIL",
    "HbCS": "HBCS",
    "A32427": "HBCS",
    "HbQs": "HBQS",
    "A32441": "HBQS",
    "2delT": "C2DELT",
    "A32429": "C2DELT",
    
    "-28": "28",
    "cd 27/28": "28",
    "CD28": "28",
    "cd15": "CD15",
    "cd 17": "CD17",
    "CD17": "CD17",
    "A32435": "CD17",
    "Co17": "CD17",
    "cd 19": "CD19",
    "cd 8/9": "CD8.9",
    "cd 26": "CD26",
    "CD26": "CD26",
    "Co26": "CD26",
    "A32433": "CD26",
    "41/42": "CD41.42",
    "A32442": "CD41.42",
    "cd 43": "CD43",
    "71/72": "CD71.72",
    "cd 95": "CD95",
    "CD95": "CD95",
    "cd121": "CD121",
    "IVS": "IVS1.1",
    "A32440": "IVS2.654",
    "89/90": "90",
    "cd90": "90",
    "-88": "88",
    "b-29": "29",
    "b-31": "31"
}

pattern = {e: None for e in mp.keys()}
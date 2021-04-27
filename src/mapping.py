feature_names = ['750HbA1', '751HbA2', '752HbE', '753HbF']

alpha_names = ['431SEA', '802a-SEA', '965Tha.SEA', 'A32431', '810a-3.7', '816a-anti-3.7', '969Tha.Alpha3.7', '428a3.7', 'A32428', '426a4.2', '805a-4.2', '968Tha.Alpha4.2', 'A32426', '432THAI', '813C', '966Tha.THAI', 'A32432', '809a-FIL', '9873FIL', '967Tha.FIL', '427HbCS', '804a-HbCs', '9850HbCs', 'A32427', '803a-HbQs', '9875HbQs', '441HbQS', 'A32441', '429c.2delT', '9858c.2delT', 'A32429']

beta_names = ['9856-28','937Tha.Co17', '830b-28', '836b-cd 27/28', '9830b-28', '9836b-cd 27/28', '438-28', '934Tha.CD28', '9842b-cd15', '824b-cd 17', '9824b-cd 17', '435CD17', '937Tha.CD17', 'A32435', '9835b-cd 19', '9838b-cd 8/9', '827b-cd 26', '9827b-cd 26', '433CD26', '963Tha.Co26', 'A32433', '931Tha.CD41/42', '825b-cd 41/42', '9825b-cd 41/42', '442cd41/42', 'A32442', '9839b-cd 43 G', '839b-cd 43 G', '829b-cd71/72', '9829b-cd71/72', '436CD71/72', '932Tha.CD71/72', '828b-cd 95', '9828b-cd 95', '936Tha.CD95', '9846b-cd121', '818a-IVS 1', '832b-IVS I.1', '9832b-IVS I.1', '439IVSI-1', '970Tha.IVSI1', '831b-IVS I.5', '9831b-IVS I.5', '964Tha. IVSI5', '9845b-IVS 2.1', '9833b-IVS II654', '833b-IVS II654', '440IVSII654', '9874IVSII654', 'A32440', '843b-cd89/90', '844b-cd90 G', '9843b-cd89/90', '9844b-cd90 G', '876-88', '9834b-29', '9840b-31']
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

 

    "431SEA": "SEA",
    "802a-SEA": "SEA",
    "965Tha.SEA": "SEA",
    "A32431": "SEA",
    "810a-3.7": "A37",
    "816a-anti-3.7": "A37",
    "969Tha.Alpha3.7":"A37",
    "428a3.7": "A37",
    "A32428": "A37",
    "426a4.2": "A42",
    "805a-4.2": "A42",
    "968Tha.Alpha4.2": "A42",
    "A32426": "A42",
    "432THAI": "THAI",
    "813C": "THAI",
    "966Tha.THAI": "THAI",
    "A32432": "THAI",
    "809a-FIL": "FIL",
    "9873FIL": "FIL",
    "967Tha.FIL": "FIL",
    "427HbCS": "HBCS",
    "804a-HbCs": "HBCS",
    "9850HbCs": "HBCS",
    "A32427": "HBCS",
    "803a-HbQs": "HBQS",
    "9875HbQs": "HBQS",
    "441HbQS": "HBQS",
    "A32441": "HBQS",
    "429c.2delT": "C2DELT",
    "9858c.2delT": "C2DELT",
    "A32429": "C2DELT",
    
    "830b-28": "28",
    "836b-cd 27/28": "28",
    "9830b-28": "28",
    "9836b-cd 27/28": "28",
    "438-28": "28",
    "9856-28": "28",
    "934Tha.CD28": "28",
    "9842b-cd15": "CD15",
    "824b-cd 17": "CD17",
    "9824b-cd 17": "CD17",
    "435CD17": "CD17",
    "937Tha.CD17": "CD17",
    "A32435": "CD17",
    "937Tha.Co17": "CD17",
    "9835b-cd 19": "CD19",
    "9838b-cd 8/9": "CD8.9",
    "827b-cd 26": "CD26",
    "9827b-cd 26": "CD26",
    "433CD26": "CD26",
    "963Tha.Co26": "CD26",
    "A32433": "CD26",
    "931Tha.CD41/42": "CD41.42",
    "825b-cd 41/42": "CD41.42",
    "9825b-cd 41/42": "CD41.42",
    "442cd41/42": "CD41.42",
    "A32442": "CD41.42",
    "9839b-cd 43 G": "CD43",
    "839b-cd 43 G": "CD43",
    "829b-cd71/72": "CD71.72",
    "9829b-cd71/72": "CD71.72",
    "436CD71/72": "CD71.72",
    "932Tha.CD71/72": "CD71.72",
    "828b-cd 95": "CD95",
    "9828b-cd 95": "CD95",
    "936Tha.CD95": "CD95",
    "9846b-cd121": "CD121",
    "818a-IVS 1": "IVS1.1",
    "832b-IVS I.1": "IVS1.1",
    "9832b-IVS I.1": "IVS1.1",
    "439IVSI-1": "IVS1.1",
    "970Tha.IVSI1": "IVS1.1",
    "831b-IVS I.5": "IVS1.5",
    "9831b-IVS I.5": "IVS1.5",
    "964Tha. IVSI5": "IVS1.5",
    "9845b-IVS 2.1": "IVS2.1",
    "9833b-IVS II654": "IVS2.654",
    "833b-IVS II654": "IVS2.654",
    "440IVSII654": "IVS2.654",
    "9874IVSII654": "IVS2.654",
    "A32440": "IVS2.654",
    "843b-cd89/90": "90",
    "844b-cd90 G": "90",
    "9843b-cd89/90": "90",
    "9844b-cd90 G": "90",
    "876-88": "88",
    "9834b-29": "29",
    "9840b-31": "31"
}

pattern = {e: None for e in mp.keys()}
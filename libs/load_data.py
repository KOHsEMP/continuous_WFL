import os
import pandas as pd
from scipy.io.arff import loadarff 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def load_data(data_name, data_path, sample_size, seed=42):
    '''
    Load data
    Args:
        data_name: dataset name (choices: 'adult', 'bank')
        data_path: path storing dataset
        sample_size: using sample size. if sample_size < 0, then all data is used. otherwise, some samples are sampled ramdomly
        seed: random seed
    Returns:
        data_df: pd.DataFrame
        cat_cols: list of categorical feature names
    '''
    
    if data_name == 'diabetes':
        data_df = pd.read_csv(os.path.join(data_path, data_name, 'diabetic_data.csv'))
        data_df = data_df.drop(['weight', 'max_glu_serum', 'A1Cresult', 'medical_specialty', 'payer_code'], axis=1) # too many missing value
        data_df = data_df.drop(['diag_3', 'diag_2', 'diag_1'], axis=1) # contains float and str and too many uniq values
        data_df = data_df.drop(['encounter_id', 'patient_nbr'], axis=1) # drop id
        data_df = data_df.drop(['citoglipton', 'examide'], axis=1) # these cols have only 1 value.

        data_df = data_df.drop(['metformin-pioglitazone', 'metformin-rosiglitazone', 'glimepiride-pioglitazone', 
                                'troglitazone', 'acetohexamide', 
                                ], axis=1) # super imbalance cols
        
        # temporary not included cols: ['glipizide-metformin', 'glyburide-metformin', 'tolazamide', 'miglitol', 'acarbose', 'tolbutamide', 'chlorpropamide' ]
        data_df = data_df.loc[(data_df['glyburide-metformin']!='Up') & (data_df['glyburide-metformin']!='Down')].reset_index(drop=True)
        data_df = data_df.loc[data_df['tolazamide'] != 'Up'].reset_index(drop=True)
        data_df = data_df.loc[(data_df['miglitol']!='Down') & (data_df['miglitol']!='Up')].reset_index(drop=True)
        data_df = data_df.loc[(data_df['acarbose']!='Down') & (data_df['acarbose']!='Up')].reset_index(drop=True)
        data_df = data_df.loc[(data_df['chlorpropamide']!='Down') & (data_df['chlorpropamide']!='Up')].reset_index(drop=True)

        data_df = data_df.loc[data_df['gender'] != 'Unknown/Invalid'].reset_index(drop=True) # There are 3 samples of 'Unknown/Invalid' -> drop

        cat_cols = ['race', 'age', 'number_diagnoses', 
                    'insulin', 'rosiglitazone', 'pioglitazone', 'glyburide', 'glipizide', 'glimepiride','nateglinide', 'repaglinide',
                    'metformin', 
                    ]
        le_cols = ['gender', 'change', 'diabetesMed', 'readmitted',
                   'glipizide-metformin', 'glyburide-metformin', 'tolazamide', 'miglitol', 'acarbose', 'tolbutamide', 'chlorpropamide']

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
        
        # decide target from 'change', 'diabetesMed', 'readmitted'
        data_df.rename(columns={'readmitted':'target'}, inplace=True)

    elif data_name == "adult":
        data_df = pd.read_csv(os.path.join(data_path, data_name, "adult.data"), header=None)
        data_df.rename(columns={0:"age", 1:"workclass", 2:"fnlwgt", 3:"education", 4:"education-num",
                                5:"marital-status", 6:"occupation", 7:"relationship", 8:"race", 9:"sex",
                                10:"capital-gain", 11:"capital-loss", 12:"hours-per-week", 13:"native-country",
                                14:"target"},
                        inplace=True)
        cat_cols = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "native-country"]

        #data_df = data_df[cat_cols + ['target', 'sex']]

        # delete rows that have missing values
        data_df = data_df.dropna(how='any')
        # label encoding
        le = LabelEncoder()
        le_cols = ['sex', 'target']
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
    
    elif data_name == "bank":
        data_df = pd.read_csv(os.path.join(data_path, data_name, "bank-full.csv"), sep=';')
        data_df = data_df.rename(columns={"y":"target"})
        data_df = data_df.drop_duplicates().reset_index(drop=True) 

        def month2num(x):
            return str(x).replace('jan','1').replace('feb', '2').replace('mar', '3').replace('apr', '4').replace('may','5').replace('jun', '6').replace('jul', '7').replace('aug', '8').replace('sep', '9').replace('oct','10').replace('nov', '11').replace('dec', '12')

        data_df['month'] = data_df['month'].map(month2num).astype(int)

        cat_cols = ['job', 'marital', 'education', 'contact', 'poutcome']
        le_cols = ['default', 'housing', 'loan', 'target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])
        
        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == "default": 
        data_df = pd.read_excel(os.path.join(data_path, data_name, "default.xls"))
        data_df.columns = data_df.iloc[0]
        data_df = data_df.drop(data_df.index[0])
        data_df.reset_index(drop=True, inplace=True)
        
        data_df = data_df.drop(['ID'], axis=1)
        data_df = data_df.rename(columns={'default payment next month': 'target'})

        cat_cols = ['EDUCATION', 'MARRIAGE']
        le_cols = ['target', 'SEX'] 
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

        # for continuous experiments
        data_df = data_df.drop(['EDUCATION', 'MARRIAGE', 'SEX'], axis=1)
        cat_cols = []

    elif data_name == 'kick':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'kick.arff'))[0])
        data_df = data_df.rename(columns={'IsBadBuy':'target'})
        data_df = data_df.drop(['WheelTypeID'], axis=1)
        data_df = data_df.dropna(how='any', axis=0) 

        for col in ['BYRNO', 'VNZIP1']:
            data_df[col] = data_df[col].map(lambda x: int(x.decode())) # b'string' -> int

        cat_cols = ['Auction', 'Make', 'Model', 'Trim', 'SubModel', 'Color', 'WheelType', 'Nationality',
                    'Size', 'TopThreeAmericanName', 'AUCGUART', 'VNST']

        le_cols = ['target', 'IsOnlineSale', 
                    'PRIMEUNIT', 'Transmission'] # by deleting samples

        # del few patterns
        for cat_col in le_cols + cat_cols:
            few_sample_list = []
            uniq_dict = data_df[cat_col].value_counts().to_dict()
            for uniq_val, num in uniq_dict.items():
                if num < 50:
                    few_sample_list.append(uniq_val)
            data_df = data_df.loc[~data_df[cat_col].isin(few_sample_list)]
            data_df.reset_index(drop=True, inplace=True)

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])
        for le_col in cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
    
    elif data_name == 'census':
        data_df = pd.read_csv(os.path.join(data_path, data_name, 'census-income.data'), header=None,
                            names=['AAGE', 'ACLSWKR', 'ADTINK', 'ADTOCC', 'AHGA', 'AHSCOL', 'AMARITL',
                                    'AMJIND', 'AMJOCC', 'ARACE', 'AREORGN', 'ASEX', 'AUNMEM', 'AUNTYPE', 'AWKSTAT',
                                    'CAPGAIN', 'GAPLOSS', 'DIVVAL', 'FILESTAT', 'GRINREG', 'GRINST', 'HHDFMX',
                                    'HHDREL', 'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'MIGSUN', 'NOEMP',
                                    'PARENT', 'PEFNTVTY', 'PEMNTVTY', 'PENATVTY', 'PRCITSHP', 'SEOTR', 'VETQVA', 'VETYYN', 
                                    'WKSWORK', 'income'])
        data_df = data_df.rename(columns={'income':'target'})

        cat_cols = ['ADTINK', 'AHGA', 'AHSCOL', 'AMARITL', 'AMJIND', 'AMJOCC', 'ARACE', 'ASEX', 'AUNMEM', 'AUNTYPE', 'DIVVAL', 'FILESTAT', 
                    'GRINST', 'HHDFMX', 'MARSUPWT', 'MIGMTR1', 'MIGMTR3', 'MIGMTR4', 'MIGSAME', 'NOEMP', 'PENATVTY', 'PEMNTVTY']
        le_cols = ['target', 'AREORGN', 'VETQVA', 'WKSWORK']

        drop_cols = ['GRINREG', 'PARENT', 'PEFNTVTY', 'SEOTR'] 

        data_df = data_df.drop(drop_cols, axis=1)

        # del few patterns
        for cat_col in cat_cols + le_cols:
            few_sample_list = []
            uniq_dict = data_df[cat_col].value_counts().to_dict()
            for uniq_val, num in uniq_dict.items():
                if num < 500:
                    few_sample_list.append(uniq_val)
            data_df = data_df.loc[~data_df[cat_col].isin(few_sample_list)]
            data_df.reset_index(drop=True, inplace=True)

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols + cat_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == 'run-or-walk':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'run-or-walk.arff'))[0])
        data_df.rename(columns={'activity': 'target'}, inplace=True)

        cat_cols = []
        le_cols = ['target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == 'jets':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'hls4ml_HLF.arff'))[0])
        data_df.rename(columns={'class': 'target'}, inplace=True)

        cat_cols = []
        le_cols = ['target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == 'jannis':
        data_df = pd.read_csv(os.path.join(data_path, data_name, 'jannis.txt'), header=None)
        data_df.rename(columns={54: 'target'}, inplace=True)

        rename_dict = {}
        for i in range(54):
            rename_dict[i] = str(i)
        data_df.rename(columns=rename_dict, inplace=True)

        cat_cols = []
        le_cols = ['target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == 'higgs':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'higgs.arff'))[0])
        data_df.rename(columns={'class': 'target'}, inplace=True)

        data_df = data_df.drop_duplicates().reset_index(drop=True)
        data_df = data_df.dropna(how='any').reset_index(drop=True)


        cat_cols = []
        le_cols = ['target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])
    
    elif data_name == 'numerai':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'numerai.arff'))[0])
        data_df.rename(columns={'attribute_21': 'target'}, inplace=True)

        cat_cols = []
        le_cols = ['target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == 'electricity':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'electricity-normalized.arff'))[0])
        data_df.rename(columns={'class': 'target'}, inplace=True)
        data_df = data_df.drop(['date'], axis=1) 

        cat_cols = []
        le_cols = ['target']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    elif data_name == 'mv':
        data_df = pd.DataFrame(loadarff(os.path.join(data_path, data_name, 'mv.arff'))[0])
        data_df.rename(columns={'binaryClass': 'target'}, inplace=True)

        cat_cols = ['x3']
        le_cols = ['target', 'x7', 'x8']
        num_cols = list(set(data_df.columns.tolist()) - set(cat_cols) - set(le_cols))

        # label encoding
        le = LabelEncoder()
        for le_col in le_cols:
            data_df[le_col] = le.fit_transform(data_df[le_col])

        # normalization
        scaler = MinMaxScaler()
        data_df[num_cols] = scaler.fit_transform(data_df[num_cols])

    else:
        raise NotImplementedError
    
    if sample_size > 0:
        data_df = data_df.sample(n=sample_size, random_state=seed)
        data_df.reset_index(drop=True, inplace=True)

    return data_df, num_cols, cat_cols




def weak_cols_code(dataset_name, weak_cols):
    if dataset_name == 'adult':
        num_cols = ['capital-gain', 'hours-per-week', 'fnlwgt', 'capital-loss', 'age', 'education-num']
    elif dataset_name == 'diabetes':
        num_cols = ['admission_source_id', 'admission_type_id', 'number_outpatient', 'num_medications', 'number_inpatient', 'number_emergency', 'discharge_disposition_id', 'time_in_hospital', 'num_lab_procedures', 'num_procedures']
    elif dataset_name == 'default':
        num_cols = ['PAY_AMT2', 'BILL_AMT2', 'PAY_AMT3', 'AGE', 'PAY_AMT1', 'BILL_AMT3', 'PAY_6', 'BILL_AMT5', 'PAY_4', 'PAY_3', 'PAY_AMT4', 'PAY_AMT6', 'PAY_2', 'BILL_AMT1', 'LIMIT_BAL', 'PAY_AMT5', 'BILL_AMT4', 'PAY_5', 'PAY_0', 'BILL_AMT6']
    elif dataset_name == 'run-or-walk':
        num_cols = ['gyro_x', 'acceleration_z', 'gyro_y', 'gyro_z', 'acceleration_y', 'acceleration_x']
    elif dataset_name == 'jets':
        num_cols = ['m2_b1_mmdt', 'm2_b2_mmdt', 'd2_a1_b1_mmdt', 'c1_b2_mmdt', 'multiplicity', 'c1_b1_mmdt', 'n2_b1_mmdt', 'n2_b2_mmdt', 'zlogz', 'c2_b1_mmdt', 'd2_a1_b2_mmdt', 'd2_b1_mmdt', 'mass_mmdt', 'c1_b0_mmdt', 'c2_b2_mmdt', 'd2_b2_mmdt']
    elif dataset_name == 'jannis':
        num_cols = ['10', '35', '26', '34', '3', '33', '31', '12', '30', '45', '29', '21', '41', '28', '0', '1', '4', '16', '32', '49', '46', '53', '42', '13', '27', '20', '38', '2', '8', '22', '37', '9', '14', '52', '51', '50', '43', '48', '5', '19', '11', '24', '40', '47', '6', '18', '15', '36', '7', '23', '17', '39', '25', '44']
    elif dataset_name == 'higgs':
        num_cols = ['jet2b-tag', 'lepton_pT', 'm_jj', 'm_bb', 'jet4b-tag', 'jet2phi', 'jet1b-tag', 'jet4pt', 'jet4phi', 'jet3pt', 'jet3eta', 'jet4eta', 'm_jjj', 'jet2eta', 'm_wwbb', 'lepton_phi', 'jet1eta', 'lepton_eta', 'jet1phi', 'missing_energy_phi', 'jet3b-tag', 'missing_energy_magnitude', 'jet2pt', 'm_wbb', 'jet1pt', 'jet3phi', 'm_lv', 'm_jlv']
    elif dataset_name == 'numerai':
        num_cols = [f"attribute_{i}" for i in range(21)]
    elif dataset_name == 'electricity':
        num_cols = ['nswprice', 'vicprice', 'day', 'vicdemand', 'period', 'nswdemand', 'transfer']
    elif dataset_name == 'mv':
        num_cols = ['x9', 'x2', 'x10', 'x6', 'x4', 'x5', 'x1']
    
    code = ""
    for num_col in num_cols:
        if num_col in weak_cols:
            code += "1"
        else:
            code += "0"
            
    code = int(code, 2)
    
    return code
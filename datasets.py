import numpy as np
import pandas as pd

# Constants
PATH = '~/Projects/datasets/data/'

#
def load_bcw(dropna=True, verbosity=False):    
    ''' 
        Breast Cancer Winsconsin 
        
        COLUMNS
        -------------------------------------------
            ID*
            Clump Thickness
            Uniformity of Cell Size
            Uniformity of Cell Shape
            Marginal Adhesion
            Single Epithelial Cell Size
            Bare Nuclei**
            Bland Chromatin
            Normal Nucleoli
            Mitoses
            Class

            * Delete
            ** Drop NaN
        
        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data
    '''
    
    NAME = 'bcw.data'
    COLUMNS = [
        'ID',
        'Clump Thickness',
        'Uniformity of Cell Size',
        'Uniformity of Cell Shape',
        'Marginal Adhesion',
        'Single Epithelial Cell Size',
        'Bare Nuclei',
        'Bland Chromatin',
        'Normal Nucleoli',
        'Mitoses',
        'Class'
    ]
    
    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)

    df.drop(['ID'], axis=1, inplace=True)
    df.replace('?', np.NaN, inplace=True)
    df.dropna(inplace=True)
    df['Bare Nuclei'] = df['Bare Nuclei'].astype('int')
    
    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_car(dropna=True, verbosity=False):
    ''' 
        Car Evaluation Database
        Marko Bohanec

        COLUMNS
        -------------------------------------------
            Buying
            Maint
            Doors
            Persons
            Luggage Boot
            Safety
        
        Note: All data items was categorized

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data
    '''

    NAME = 'car.data'
    COLUMNS = [
        'Buying',
        'Maint',
        'Doors',
        'Persons',
        'Luggage Boot',
        'Safety',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)

    for c in df.columns:
        df[c] = pd.Categorical(df[c]).codes


    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_cortex_nuclear(dropna=True, verbosity=False):
    ''' 
        Data Cortex Nuclear

        COLUMNS
        -------------------------------------------
            MouseID*,
            DYRK1A_N,
            ITSN1_N,
            BDNF_N,
            NR1_N,
            NR2A_N,
            pAKT_N,
            pBRAF_N,
            pCAMKII_N,
            pCREB_N,
            pELK_N,
            pERK_N,
            pJNK_N,
            PKCA_N,
            pMEK_N,
            pNR1_N,
            pNR2A_N,
            pNR2B_N,
            pPKCAB_N,
            pRSK_N,
            AKT_N,
            BRAF_N,
            CAMKII_N,
            CREB_N,
            ELK_N,
            ERK_N,
            GSK3B_N,
            JNK_N,
            MEK_N,
            TRKA_N,
            RSK_N,
            APP_N,
            Bcatenin_N,
            SOD1_N,
            MTOR_N,
            P38_N,
            pMTOR_N,
            DSCR1_N,
            AMPKA_N,
            NR2B_N,
            pNUMB_N,
            RAPTOR_N,
            TIAM1_N,
            pP70S6_N,
            NUMB_N,
            P70S6_N,
            pGSK3B_N,
            pPKCG_N,
            CDK5_N,
            S6_N,
            ADARB1_N,
            AcetylH3K9_N,
            RRP1_N,
            BAX_N,
            ARC_N,
            ERBB4_N,
            nNOS_N,
            Tau_N,
            GFAP_N,
            GluR3_N,
            GluR4_N,
            IL1B_N,
            P3525_N,
            pCASP9_N,
            PSD95_N,
            SNCA_N,
            Ubiquitin_N,
            pGSK3B_Tyr216_N,
            SHH_N,
            BAD_N,
            BCL2_N,
            pS6_N,
            pCFOS_N,
            SYP_N,
            H3AcK18_N,
            EGR1_N,
            H3MeK4_N,
            CaNA_N,
            Genotype,
            Treatment,
            Behavior,
            Class
            
            *Delete

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls
    '''

    NAME = 'cortex_nuclear.data'
    COLUMNS = [
        'MouseID',
        'DYRK1A_N',
        'ITSN1_N',
        'BDNF_N',
        'NR1_N',
        'NR2A_N',
        'pAKT_N',
        'pBRAF_N',
        'pCAMKII_N',
        'pCREB_N',
        'pELK_N',
        'pERK_N',
        'pJNK_N',
        'PKCA_N',
        'pMEK_N',
        'pNR1_N',
        'pNR2A_N',
        'pNR2B_N',
        'pPKCAB_N',
        'pRSK_N',
        'AKT_N',
        'BRAF_N',
        'CAMKII_N',
        'CREB_N',
        'ELK_N',
        'ERK_N',
        'GSK3B_N',
        'JNK_N',
        'MEK_N',
        'TRKA_N',
        'RSK_N',
        'APP_N',
        'Bcatenin_N',
        'SOD1_N',
        'MTOR_N',
        'P38_N',
        'pMTOR_N',
        'DSCR1_N',
        'AMPKA_N',
        'NR2B_N',
        'pNUMB_N',
        'RAPTOR_N',
        'TIAM1_N',
        'pP70S6_N',
        'NUMB_N',
        'P70S6_N',
        'pGSK3B_N',
        'pPKCG_N',
        'CDK5_N',
        'S6_N',
        'ADARB1_N',
        'AcetylH3K9_N',
        'RRP1_N',
        'BAX_N',
        'ARC_N',
        'ERBB4_N',
        'nNOS_N',
        'Tau_N',
        'GFAP_N',
        'GluR3_N',
        'GluR4_N',
        'IL1B_N',
        'P3525_N',
        'pCASP9_N',
        'PSD95_N',
        'SNCA_N',
        'Ubiquitin_N',
        'pGSK3B_Tyr216_N',
        'SHH_N',
        'BAD_N',
        'BCL2_N',
        'pS6_N',
        'pCFOS_N',
        'SYP_N',
        'H3AcK18_N',
        'EGR1_N',
        'H3MeK4_N',
        'CaNA_N',
        'Genotype',
        'Treatment',
        'Behavior',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df.replace('?', np.NaN, inplace=True)
    df.drop('MouseID', axis=1, inplace=True)

    df['Genotype'] = df['Genotype'].astype('category').cat.codes
    df['Treatment'] = df['Treatment'].astype('category').cat.codes
    df['Behavior'] = df['Behavior'].astype('category').cat.codes
    df['Class'] = df['Class'].astype('category').cat.codes

    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_credit_card_clients(dropna=True, verbosity=False):    
    ''' 
        Credit Card Clients

        COLUMNS
        -------------------------------------------
            ID*,
            Limit_bal,
            Sex,
            Education,
            Marriage,
            Age,
            Pay_0,
            Pay_2,
            Pay_3,
            Pay_4,
            Pay_5,
            Pay_6,
            Bill_amt1,
            Bill_amt2,
            Bill_amt3,
            Bill_amt4,
            Bill_amt5,
            Bill_amt6,
            Pay_amt1,
            Pay_amt2,
            Pay_amt3,
            Pay_amt4,
            Pay_amt5,
            Pay_amt6,
            Class

            * Delete
            
        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
    '''

    NAME = 'credit_card_clients.data'
    COLUMNS = [
        'ID',
        'Limit_bal',
        'Sex',
        'Education',
        'Marriage',
        'Age',
        'Pay_0',
        'Pay_2',
        'Pay_3',
        'Pay_4',
        'Pay_5',
        'Pay_6',
        'Bill_amt1',
        'Bill_amt2',
        'Bill_amt3',
        'Bill_amt4',
        'Bill_amt5',
        'Bill_amt6',
        'Pay_amt1',
        'Pay_amt2',
        'Pay_amt3',
        'Pay_amt4',
        'Pay_amt5',
        'Pay_amt6',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)

    df.drop(['ID'], axis=1, inplace=True)
    
    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_dermatology(dropna=True, verbosity=False):    
    ''' 
        Dermatology Dataset

        COLUMNS
        -------------------------------------------
            Erythema,
            Scaling,
            Definite Borders,
            Itching,
            Koebner Phenomenon,
            Polygonal Papules,
            Follicular Papules,
            Oral Mucosal Involvement,
            Knee And Elbow Involvement,
            Scalp Involvement,
            Family History,
            Melanin Incontinence,
            Eosinophils In The Infiltrate,
            Pnl Infiltrate,
            Fibrosis Of The Papillary Dermis,
            Exocytosis,
            Acanthosis,
            Hyperkeratosis,
            Parakeratosis,
            Clubbing Of The Rete Ridges,
            Elongation Of The Rete Ridges,
            Thinning Of The Suprapapillary Epidermis,
            Spongiform Pustule,
            Munro Microabcess,
            Focal Hypergranulosis,
            Disappearance Of The Granular Layer,
            Vacuolisation And Damage Of Basal Layer,
            Spongiosis,
            Saw-Tooth Appearance Of Retes,
            Follicular Horn Plug,
            Perifollicular Parakeratosis,
            Inflammatory Monoluclear Inflitrate,
            Band-Like Infiltrate,
            Age**,
            Class

            **Missing Values

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data
    '''

    NAME = 'dermatology.data'
    COLUMNS = [
        'Erythema',
        'Scaling',
        'Definite Borders',
        'Itching',
        'Koebner Phenomenon',
        'Polygonal Papules',
        'Follicular Papules',
        'Oral Mucosal Involvement',
        'Knee And Elbow Involvement',
        'Scalp Involvement',
        'Family History',
        'Melanin Incontinence',
        'Eosinophils In The Infiltrate',
        'Pnl Infiltrate',
        'Fibrosis Of The Papillary Dermis',
        'Exocytosis',
        'Acanthosis',
        'Hyperkeratosis',
        'Parakeratosis',
        'Clubbing Of The Rete Ridges',
        'Elongation Of The Rete Ridges',
        'Thinning Of The Suprapapillary Epidermis',
        'Spongiform Pustule',
        'Munro Microabcess',
        'Focal Hypergranulosis',
        'Disappearance Of The Granular Layer',
        'Vacuolisation And Damage Of Basal Layer',
        'Spongiosis',
        'Saw-Tooth Appearance Of Retes',
        'Follicular Horn Plug',
        'Perifollicular Parakeratosis',
        'Inflammatory Monoluclear Inflitrate',
        'Band-Like Infiltrate',
        'Age',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df.replace('?', np.NaN, inplace=True)

    if dropna:
        df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

#
def load_ecoli(dropna=True, verbosity=False):
    ''' 
        Ecoli Data Set

        COLUMNS
        -------------------------------------------
            Sequence Name*,
            MCG,
            GVH,
            LIP,
            CHG,
            AAC,
            ALM1,
            ALM2,
            Class,
            
            *Delete

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data
    '''

    NAME = 'ecoli.data'
    COLUMNS = [
        'Sequence Name',
        'MCG',
        'GVH',
        'LIP',
        'CHG',
        'AAC',
        'ALM1',
        'ALM2',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df.replace('?', np.NaN, inplace=True)
    df.drop('Sequence Name', axis=1, inplace=True)

    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_eeg_eye_state(dropna=True, verbosity=False):
    ''' 
        Data Cortex Nuclear

        COLUMNS
        -------------------------------------------
            AF3,
            F7,
            F3,
            FC5,
            T7,
            P7,
            O1,
            O2,
            P8,
            T8,
            FC6,
            F4,
            F8,
            AF4,
            Class


        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff
    '''

    NAME = 'eeg_eye_state.data'
    COLUMNS = [
        'AF3',
        'F7',
        'F3',
        'FC5',
        'T7',
        'P7',
        'O1',
        'O2',
        'P8',
        'T8',
        'FC6',
        'F4',
        'F8',
        'AF4',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)

    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')
        
    return df

#
def load_glass(dropna=True, verbosity=False):    
    ''' 
        Glass Type Dataset
        
        COLUMNS
        -------------------------------------------
            ID*,
            Refractive Index,
            Na,
            Mg,
            Al,
            Si,
            K,
            Ca,
            Ba,
            Fe,
            Class

            * Delete
        
        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data
    '''
    
    NAME = 'glass.data'
    COLUMNS = [
        'ID',
        'Refractive Index',
        'Na',
        'Mg',
        'Al',
        'Si',
        'K',
        'Ca',
        'Ba',
        'Fe',
        'Class'
    ]
    
    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    
    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_haberman(dropna=True, verbosity=False):
    ''' 
        Haberman's Survival Data Set

        COLUMNS
        -------------------------------------------
            Age,
            Years of Operation,
            Positive Axillary Nodes,
            Class


        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff
    '''

    NAME = 'haberman.data'
    COLUMNS = [
        'Age',
        'Years of Operation',
        'Positive Axillary Nodes',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)

    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_ionosphere(dropna=True, verbosity=False):
    ''' 
        Ionosphere Data Set

        COLUMNS
        -------------------------------------------
            ATT 1,
            ATT 2,
            ATT 3,
            ATT 4,
            ATT 5,
            ATT 6,
            ATT 7,
            ATT 8,
            ATT 9,
            ATT 10,
            ATT 11,
            ATT 12,
            ATT 13,
            ATT 14,
            ATT 15,
            ATT 16,
            ATT 17,
            ATT 18,
            ATT 19,
            ATT 20,
            ATT 21,
            ATT 22,
            ATT 23,
            ATT 24,
            ATT 25,
            ATT 26,
            ATT 27,
            ATT 28,
            ATT 29,
            ATT 30,
            ATT 31,
            ATT 32,
            ATT 33,
            ATT 34,
            Class

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
    '''

    NAME = 'ionosphere.data'
    COLUMNS = [
        'ATT 1',
        'ATT 2',
        'ATT 3',
        'ATT 4',
        'ATT 5',
        'ATT 6',
        'ATT 7',
        'ATT 8',
        'ATT 9',
        'ATT 10',
        'ATT 11',
        'ATT 12',
        'ATT 13',
        'ATT 14',
        'ATT 15',
        'ATT 16',
        'ATT 17',
        'ATT 18',
        'ATT 19',
        'ATT 20',
        'ATT 21',
        'ATT 22',
        'ATT 23',
        'ATT 24',
        'ATT 25',
        'ATT 26',
        'ATT 27',
        'ATT 28',
        'ATT 29',
        'ATT 30',
        'ATT 31',
        'ATT 32',
        'ATT 33',
        'ATT 34',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df['Class'] = df['Class'].astype('category').cat.codes
        
    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_iris(dropna=True, verbosity=False):
    ''' 
        Iris Plants Database
        R.A. Fisher

        COLUMNS
        -------------------------------------------
            Sepal Length
            Sepal Width
            Petal Length
            Petal Width
            Class
            0. Iris-setosa: 50
            1. Iris-versicolor: 50
            2. Iris-virginica: 50
        
        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data
    '''

    NAME = 'iris.data'
    COLUMNS = [
        'Sepal Length',
        'Sepal Width',
        'Petal Length',
        'Petal Width',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df['Class'] = df['Class'].astype('category').cat.codes

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_messidor(dropna=True, verbosity=False):
    ''' 
        Diabetic Retinopathy Debrecen

        COLUMNS
        -------------------------------------------
            Quality assessment,
            Pre-screening,
            MA Detection 0.5,
            MA Detection 0.6,
            MA Detection 0.7,
            MA Detection 0.8,
            MA Detection 0.9,
            MA Detection 1.0,
            MA detection Exut 1,
            MA detection Exut 2,
            MA detection Exut 3,
            MA detection Exut 4,
            MA detection Exut 5,
            MA detection Exut 6,
            MA detection Exut 7,
            MA detection Exut 8,
            Distance,
            Diameter,
            AmFm Classification,
            Class,

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff
    '''

    NAME = 'messidor_features.data'
    COLUMNS = [
        'Quality Assessment',
        'Pre-screening',
        'MA Detection 0.5',
        'MA Detection 0.6',
        'MA Detection 0.7',
        'MA Detection 0.8',
        'MA Detection 0.9',
        'MA Detection 1.0',
        'MA detection Exut 1',
        'MA detection Exut 2',
        'MA detection Exut 3',
        'MA detection Exut 4',
        'MA detection Exut 5',
        'MA detection Exut 6',
        'MA detection Exut 7',
        'MA detection Exut 8',
        'Distance',
        'Diameter',
        'AmFm Classification',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df.replace('?', np.NaN, inplace=True)

    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_nursery(dropna=True, verbosity=False):
    ''' 
        Nursery Data Set

        COLUMNS
        -------------------------------------------
            Parents
            Has Nurs
            Form
            Children
            Housing
            Finance
            Social
            Health
            Class

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data
    '''

    NAME = 'nursery.data'
    COLUMNS = [
        'Parents',
        'Has Nurs',
        'Form',
        'Children',
        'Housing',
        'Finance',
        'Social',
        'Health',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    
    for c in df.columns:
        df[c] = df[c].astype('category').cat.codes
        
    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_phishing_websites(dropna=True, verbosity=False):
    ''' 
        Phishing Websites Data Set

        COLUMNS
        -------------------------------------------
            Sfh,
            Popupwidnow,
            Sslfinal_state,
            Request_url,
            Url_of_anchor,
            Web_traffic,
            Url_length,
            Age_of_domain,
            Having_ip_address,
            Class,


        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00379/PhishingData.arff
    '''

    NAME = 'phishing.data'
    COLUMNS = [
        'Sfh',
        'Popupwidnow',
        'Sslfinal_state',
        'Request_url',
        'Url_of_anchor',
        'Web_traffic',
        'Url_length',
        'Age_of_domain',
        'Having_ip_address',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df.replace('?', np.NaN, inplace=True)

    if dropna: df.dropna(inplace=True)
        
    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_seeds(dropna=True, verbosity=False):
    ''' 
        Seeds Data Set

        COLUMNS
        -------------------------------------------
            Area,
            Perimeter,
            Compactness,
            Length of kernel,
            Width of kernel,
            Asymmetry coefficient,
            Length of kernel groove,
            Class

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt
    '''

    NAME = 'seeds.data'
    COLUMNS = [
        'Area',
        'Perimeter',
        'Compactness',
        'Length of Kernel',
        'Width of Kernel',
        'Asymmetry Coefficient',
        'Length of Kernel Groove',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', sep='\t', names=COLUMNS)
    
    for c in df.columns:
        df[c] = df[c].astype('category').cat.codes
        
    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_seismic_bumps(dropna=True, verbosity=False):
    ''' 
        Seismic Bumps Data Set

        COLUMNS
        -------------------------------------------
            Seismic,
            Seismoacoustic,
            Shift,
            Genergy,
            Gpuls,
            Gdenergy,
            Gdpuls,
            Ghazard,
            Nbumps,
            Nbumps2,
            Nbumps3,
            Nbumps4,
            Nbumps5,
            Nbumps6,
            Nbumps7,
            Nbumps89,
            Energy,
            Maxenergy,
            Class

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff
    '''

    NAME = 'seismic_bumps.data'
    COLUMNS = [
        'Seismic',
        'Seismoacoustic',
        'Shift',
        'Genergy',
        'Gpuls',
        'Gdenergy',
        'Gdpuls',
        'Ghazard',
        'Nbumps',
        'Nbumps2',
        'Nbumps3',
        'Nbumps4',
        'Nbumps5',
        'Nbumps6',
        'Nbumps7',
        'Nbumps89',
        'Energy',
        'Maxenergy',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    
    df['Seismic'] = df['Seismic'].astype('category').cat.codes
    df['Seismoacoustic'] = df['Seismoacoustic'].astype('category').cat.codes
    df['Shift'] = df['Shift'].astype('category').cat.codes
    df['Ghazard'] = df['Ghazard'].astype('category').cat.codes
    df['Class'] = df['Class'].astype('category').cat.codes
        
    if dropna: df.dropna(inplace=True)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_soybean(dropna=True, verbosity=False):
    ''' 
        Soybean Large Data Set

        COLUMNS
        -------------------------------------------
            Date,
            Plant-Stand,
            Precip,
            Temp,
            Hail,
            Crop-Hist,
            Area-Damaged,
            Severity,
            Seed-Tmt,
            Germination,
            Plant-Growth,
            Leaves,
            Leafspots-Halo,
            Leafspots-Marg,
            Leafspot-Size,
            Leaf-Shread,
            Leaf-Malf,
            Leaf-Mild,
            Stem,
            Lodging,
            Stem-Cankers,
            Canker-Lesion,
            Fruiting-Bodies,
            External Decay,
            Mycelium,
            Int-Discolor,
            Sclerotia,
            Fruit-Pods,
            Fruit Spots,
            Seed,
            Mold-Growth,
            Seed-Discolor,
            Seed-Size,
            Shriveling,
            Roots,
            Class

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-large.data
    '''

    NAME = 'soybean.data'
    COLUMNS = [
        'Class',
        'Date',
        'Plant-Stand',
        'Precip',
        'Temp',
        'Hail',
        'Crop-Hist',
        'Area-Damaged',
        'Severity',
        'Seed-Tmt',
        'Germination',
        'Plant-Growth',
        'Leaves',
        'Leafspots-Halo',
        'Leafspots-Marg',
        'Leafspot-Size',
        'Leaf-Shread',
        'Leaf-Malf',
        'Leaf-Mild',
        'Stem',
        'Lodging',
        'Stem-Cankers',
        'Canker-Lesion',
        'Fruiting-Bodies',
        'External Decay',
        'Mycelium',
        'Int-Discolor',
        'Sclerotia',
        'Fruit-Pods',
        'Fruit Spots',
        'Seed',
        'Mold-Growth',
        'Seed-Discolor',
        'Seed-Size',
        'Shriveling',
        'Roots'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df.replace('?', np.NaN, inplace=True)

    if dropna: df.dropna(inplace=True)
        
    for c in df.columns:
        df[c] = df[c].astype('category').cat.codes
        
    class_col = df.pop('Class')
    df['Class'] = class_col

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_tae(dropna=True, verbosity=False):
    ''' 
        Teaching Assistant Evaluation Data Set

        COLUMNS
        -------------------------------------------
            Native English Speaker,
            Instructor,
            Course,
            Summer/Regular,
            Class Size (numerical),
            Class


        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data
    '''

    NAME = 'tae.data'
    COLUMNS = [
        'Native English Speaker',
        'Instructor',
        'Course',
        'Summer/Regular',
        'Class Size',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df.replace('?', np.NaN, inplace=True)

    if dropna: df.dropna(inplace=True)
        
    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_wholesale(dropna=True, verbosity=False):
    ''' 
        Wholesale customers Data Set

        COLUMNS
        -------------------------------------------
            Channel**
            Region*
            Fresh
            Milk
            Grocery
            Frozen
            Detergents_Paper
            Delicassen

            *Chosen class
            **Could be class
            
        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv
    '''

    NAME = 'wholesale.data'
    COLUMNS = [
        'Channel',
        'Region',
        'Fresh,'
        'Milk',
        'Grocery',
        'Frozen',
        'Detergents_Paper',
        'Delicassen'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)
    df.replace('?', np.NaN, inplace=True)

    if dropna: df.dropna(inplace=True)
        
    df['Class'] = df.pop('Channel')
        
    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df
       
#
def load_wine(dropna=True, verbosity=False):
    ''' 
        Wine Quality 
        P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.

        COLUMNS
        -------------------------------------------
            Fixed Acidity
            Volatile Acidity
            Citric Acid
            Residual Sugar
            Chlorides
            Free Sulfur Dioxide
            Total Sulfur Dioxide
            Density
            Ph
            Sulphates
            Alcohol
            Class (Quality)

        --
        https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv
    '''

    NAME = 'wine.data'
    COLUMNS = [
        'Fixed Acidity',
        'Volatile Acidity',
        'Citric Acid',
        'Residual Sugar',
        'Chlorides',
        'Free Sulfur Dioxide',
        'Total Sulfur Dioxide',
        'Density',
        'Ph',
        'Sulphates',
        'Alcohol',
        'Class'
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS, sep=';') #, skiprows=1)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df

#
def load_titanic(dropna=True, verbosity=False):
    ''' 
        Titanic 

        COLUMNS
        -------------------------------------------
            Survived
            Pclass
            Name
            Sex
            Age
            Siblings/Spouses Aboard,
            Parents/Children Aboard
            Fare
            
            *New
        
        --
        https://www.kaggle.com/c/titanic/data?select=train.csv
    '''

    NAME = 'titanic.data'
    COLUMNS = [
        'Survived',
        'Pclass',
        'Name',
        'Sex',
        'Age',
        'SS Aboard',
        'PC Aboard',
        'Fare',
    ]

    df = pd.read_csv(f'{PATH}{NAME}', names=COLUMNS)

    df['Sex'] = df['Sex'].astype('category').cat.codes
    #df['Title'] = df['Name'].apply(lambda x: x.split('.')[0]).astype('category').cat.codes
    
    columns = list(df.columns)

    if verbosity:
        aux = '\n  '
        print(f'Data: {NAME}')
        print(f'Lines: {df.shape[0]}')
        print(f'Columns:\n  {aux.join(df.columns)}')

    return df[columns[1:] + columns[0:1]]
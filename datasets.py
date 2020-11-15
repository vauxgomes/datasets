'''
https://archive.ics.uci.edu/ml/machine-learning-databases/00329/messidor_features.arff
https://archive.ics.uci.edu/ml/machine-learning-databases/ecoli/ecoli.data
https://archive.ics.uci.edu/ml/machine-learning-databases/00264/EEG%20Eye%20State.arff
https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data
https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data
https://archive.ics.uci.edu/ml/machine-learning-databases/00342/Data_Cortex_Nuclear.xls
https://archive.ics.uci.edu/ml/machine-learning-databases/nursery/nursery.data
https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt
https://archive.ics.uci.edu/ml/machine-learning-databases/00266/seismic-bumps.arff
https://archive.ics.uci.edu/ml/machine-learning-databases/soybean/soybean-small.data
https://archive.ics.uci.edu/ml/machine-learning-databases/tae/tae.data
https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data
https://archive.ics.uci.edu/ml/machine-learning-databases/00379/PhishingData.arff
https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv
'''

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
def load_credit_card_clients(dropna=True, verbosity=False):    
    ''' 
        Glass Type Dataset

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
    Glass Type Dataset

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
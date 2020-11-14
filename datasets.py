'''

https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data
https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls
https://archive.ics.uci.edu/ml/machine-learning-databases/dermatology/dermatology.data
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
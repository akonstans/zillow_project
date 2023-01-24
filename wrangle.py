import pandas as pd
import new_lib as nl
import os
import env

def acquire_zillow():
    '''
    This function will access the zillow database from a SQL server and grab any data
    that meets the conditions of the project
    '''
    if os.path.isfile('zillow.csv'):
        zil = pd.read_csv('zillow.csv', index_col= 0)
    else:
        zil = nl.connect('zillow', 'zillow.csv', '''SELECT * FROM properties_2017 as pr
JOIN predictions_2017 AS pe ON pr.parcelid = pe.parcelid
JOIN propertylandusetype AS l ON pr.propertylandusetypeid = l.propertylandusetypeid
WHERE pe.transactiondate >= 01-01-2017 AND l.propertylandusetypeid = 261;''')
    return zil

def prep_zillow(zil):
    '''
    This function will then prepare and clean the data and make the column names more descriptive.
    It also removes any columns that are not being considered for this project
    '''
    zil = zil.rename(columns = {'bedroomcnt': 'bedrooms', 'bathroomcnt': 'bathrooms', 
                            'calculatedfinishedsquarefeet': 'square_footage', 
                            'taxvaluedollarcnt': 'tax_value', 'lotsizesquarefeet': 'lot_size'})
    zil = zil[['id', 'bedrooms', 'bathrooms', 'square_footage', 'lot_size', 'tax_value', 'yearbuilt', 'fips']]
    # getting dataframe into the right subset
    zil = zil.dropna()
    zil = zil.reset_index()
    zil = zil.drop(columns = 'index')
    zil.yearbuilt = zil.yearbuilt.astype(int)
    zil.index.name = 'index'
    return zil

def wrangle_zillow():
    '''
    A combination of the acquire and prep functions with an added function to remove any outliers from the data
    The remove outliers function comes from the custom library and is documented there as well.
    '''
    zil = prep_zillow(acquire_zillow())
    return zil
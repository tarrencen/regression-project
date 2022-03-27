import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from env import get_db_url
from pydataset import data
import os

def show_codeup_dbs():
    url = get_db_url('employees')
    codeup_dbs = pd.read_sql('SHOW DATABASES', url)
    print('List of Codeup DBs:\n')
    return codeup_dbs


def get_prop_vals():
    '''
    Returns a DataFrame composed of selected data from the properties_2017 table in the zillow database on
    Codeup's SQL server
    '''
    filename = 'prop_vals.csv'
    if os.path.exists(filename):
        print('Reading from CSV file...')
        return pd.read_csv(filename)
    query = """
    SELECT 
    bedroomcnt, 
    bathroomcnt, 
    roomcnt,
    numberofstories,
    fireplaceflag,
    poolcnt, 
    buildingqualitytypeid, 
    calculatedfinishedsquarefeet, 
    lotsizesquarefeet, 
    taxvaluedollarcnt, 
    yearbuilt, 
    taxamount, 
    fips, 
    logerror 
    FROM properties_2017 prop
    JOIN predictions_2017 pred ON pred.id = prop.id
    JOIN propertylandusetype land ON land.propertylandusetypeid = prop.propertylandusetypeid
    WHERE (
    prop.taxvaluedollarcnt IS NOT NULL
    AND 
    land.propertylandusetypeid = 261 
    AND 
    pred.transactiondate LIKE '2017%%'
    );
    """
    print('Getting a fresh copy from SQL database...')
    url = get_db_url('zillow')
    prop_vals = pd.read_sql(query, url)
    print('Copying to CSV...')
    prop_vals.to_csv(filename)
    return prop_vals



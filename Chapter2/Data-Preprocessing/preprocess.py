#!/usr/bin/env python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the UCI Machine Learning Income data from this directory

# Three different ways to read the data
# Notice this incorrectly reads the first line as the header
df0 = pd.read_csv('/data/adult.data')             

# The header=None enumerates the classes without a name 
df1 = pd.read_csv('/data/adult.data', header = None) 

# The header=None enumerates the classes without a name 
df2 = pd.read_csv('/data/adult.data', names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country','Label']) 

# Create an empty dictionary
mappings = {}

# Run through all columns in the CSV
for col_name in df2.columns:
    # If the type of variables are categorical, they will be an 'object' type
    if(df2[col_name].dtype == 'object'):
        # Create a mapping from categorical to numerical variables
        df2[col_name]= df2[col_name].astype('category')
        df2[col_name], mapping_index = pd.Series(df2[col_name]).factorize()
	# Store the mappings in dictionary
        mappings[col_name]={}
        for i in range(len(mapping_index.categories)):
             mappings[col_name][i]=mapping_index.categories[i]
    # Store a continuous tag for variables that are already numerical
    else:
        mappings[col_name] = 'continuous'

        

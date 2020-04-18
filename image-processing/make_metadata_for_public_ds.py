# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 15:53:54 2020

@author: rkako
"""
#%%
import os
import pandas as pd
import numpy as np
from datetime import datetime
import dateutil.parser

#%%
current_dir = os.getcwd()
csv_path = 'covid-chestxray-dataset/metadata.csv'
csv = pd.read_csv(csv_path)
### Only keep frontal view
idx_pa = csv["view"].isin(["PA"])
csv = csv[idx_pa]

### Make labels by combining findign and survival
Labes = [f+'_'+'O' if str(s) == 'nan' else f+'_'+str(s) for f, s in zip(csv['finding'], csv['survival'])]

### Remove unnecessary columns
cvs_columns = csv.columns
csv_unremovable_cols = ['patientid', 'sex', 'age', 'finding', 'survival',
                        'date', 'filename']

# removable_cols = ['location', 'doi', ' url', 'license', 'view', 'modality',
#                    'offset', 'clinical notes', 'other notes', 'Unnamed: 22',
#                    'finding', 'survival']

removable_cols = list(set(cvs_columns) - set(csv_unremovable_cols))
for col in removable_cols:
    csv.drop(col, axis=1, inplace=True)

### Insert label columns in the final column
num_cols = len(csv.columns)
csv.insert(num_cols, 'label', Labes)

### Convert dates from string to datetype
for ind in csv.index:
    date = csv['date'][ind]
    if pd.isnull(date):
         date = '2020'
    date_str = dateutil.parser.parse(date).strftime("%Y-%m-%d")
    csv['date'][ind] = date_str

### Remove old data 
filter_date = False
if filter_date:
    for ind in csv.index:
        date = datetime.strptime(csv['date'][ind], '%Y-%m-%d').date()
        if date.year < 2020:
            csv.drop(ind, inplace=True)        

path_prefix = 'covid-chestxray-dataset/images'
path_postfix = [path_prefix + '/' + img_name for img_name in csv['filename']]
csv.insert(6, 'path_postfix', path_postfix)

## remove classes have one point
grouped = csv.groupby(['label'])
csv = grouped.filter(lambda x: x['label'].count() > 3)

csv.to_csv('metadata_pub.csv', index=False)
import json
import os
import requests
import pandas as pd

relabel_df = pd.read_excel('imat_data/relabel.xlsx')


annotations = json.load(open('imat_data/val_annos.json', 'r'))
map_dict = dict((zip(relabel_df['labelId'], relabel_df['LabelId_new'])))

prev_label = relabel_df['labelId'].values
new_label = relabel_df['LabelId_new'].values
for annotation in annotations:
    labels = annotation['labelId']
    new_label
    for i in range(len(labels)):
        label = labels[i]
        if label in prev_label:
            index = map_dict[label]
            
        else:
            print(f'label {label} not found in relabel mapping')

with open('imat_data/val_annos_relabel.json', 'w') as f:
    json.dump(annotations, f, indent=4)
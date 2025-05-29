import json
import os
import requests
import pandas as pd

relabel_df = pd.read_excel('imat_data/relabel.xlsx')


annotations = json.load(open('imat_data/train_annos.json', 'r'))  
map_dict = dict((zip(relabel_df['labelId'], relabel_df['labelId_new'])))
print(len(annotations['annotations']))

for annotation in annotations['annotations']:
    labels = annotation['labelId']
    for i in range(len(labels)):
        label = labels[i]
        if label in map_dict:
            index = map_dict[label]
            
        else:
            print(f'label {label} not found in relabel mapping')


print(len(annotations['annotations']))
with open('imat_data/train_annos_relabel.json', 'w') as f:
    json.dump(annotations, f, indent=4)
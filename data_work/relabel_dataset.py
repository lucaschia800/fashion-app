import json
import os
import requests
import pandas as pd

relabel_df = pd.read_excel('imat_data/relabel-split.xlsx')


annotations = json.load(open('imat_data/train_annos_6-4.json', 'r'))  
map_dict = dict((zip(relabel_df['labelId'], relabel_df['labelId_new'])))
print(len(annotations['annotations']))



# print("First few mapping entries:", list(map_dict.items())[:10])
# print("Sample original labels:", annotations['annotations'][0]['labelId'][:10])

# # Check data types
# print("Map dict key type:", type(list(map_dict.keys())[0]))
# print("JSON label type:", type(annotations['annotations'][0]['labelId'][0])).

for annotation in annotations['annotations']:
    if annotation['labelId'] is None:
        print(f"Skipping annotation with imageId {annotation['imageId']} due to None labelId")
        continue
    labels = annotation['labelId']
    new_labels = []
    for i in range(len(labels)):
        label = int(labels[i])
        if label in map_dict:
            new_label = map_dict[label]
            new_labels.append(new_label)
            
        else:
            print(f'label {label} not found in relabel mapping')
    annotation['labelId'] = new_labels

print(len(annotations['annotations']))
with open('imat_data/train_annos_relabel-6-4.json', 'w') as f:
    json.dump(annotations, f, indent=4)
import json
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import time

with open('imat_data/train.json', 'r') as f:
    data = json.load(f)


print(data.keys())
annotations = data['annotations']
annotations_train = []
for i in range(len(annotations)):
    image_id = str(annotations[i]['imageId'])
    if os.path.exists(f"imat_data/img/{image_id}.jpg"):
        annotations_train.append(annotations[i])

print(f"Number of images in train set: {len(annotations_train)}")

# Save the filtered annotations to a new JSON file
output_file = 'imat_data/train_annos_6-4.json'
with open(output_file, 'w') as f:
    json.dump({'annotations': annotations_train}, f, indent=4)
    

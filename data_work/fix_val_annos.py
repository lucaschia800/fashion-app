import json

def remove_duplicate_labels(input_path, output_path):
    """Remove duplicate labels from annotations"""
    
    # Load data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Process annotations
    for annotation in data['annotations']:
        # Remove duplicates while preserving order
        annotation['labelId'] = list(dict.fromkeys(annotation['labelId']))
    
    # Save cleaned data
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Duplicates removed. Saved to: {output_path}")

# Clean both train and validation sets
remove_duplicate_labels("imat_data/val_annos_relabel.json", "imat_data/val_annos_clean.json")
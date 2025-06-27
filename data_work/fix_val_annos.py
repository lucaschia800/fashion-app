import json

def remove_duplicate_labels(input_path, output_path):
    """Remove duplicate labels from annotations"""
    
    # Load data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    # Check for duplicates and process
    duplicates_found = 0
    total_duplicates_removed = 0
    
    for annotation in data['annotations']:
        original_count = len(annotation['labelId'])
        # Remove duplicates while preserving order
        annotation['labelId'] = list(dict.fromkeys(annotation['labelId']))
        new_count = len(annotation['labelId'])
        
        if original_count != new_count:
            duplicates_found += 1
            total_duplicates_removed += (original_count - new_count)
    
    # Save cleaned data
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Found duplicates in {duplicates_found} annotations, removed {total_duplicates_removed} duplicate labels")
    print(f"Duplicates removed. Saved to: {output_path}")

# Clean both train and validation sets
remove_duplicate_labels("imat_data/train_annos_6-26.json", "imat_data/train_annos_6-26_clean.json")
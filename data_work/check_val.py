import json
import torch
import torch.nn.functional as F
from collections import Counter

def check_validation_data(json_path):
    """Check validation data for duplicate labels and other issues"""
    
    # Load the JSON data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    annotations = data['annotations']
    print(f"Total annotations: {len(annotations)}")
    print("="*60)
    
    # Statistics
    duplicate_count = 0
    empty_labels = 0
    max_duplicates = 0
    duplicate_examples = []
    
    # Check each annotation
    for i, annotation in enumerate(annotations):
        label_ids = annotation['labelId']
        image_id = annotation['imageId']
        
        # Check for empty labels
        if len(label_ids) == 0:
            empty_labels += 1
            continue
            
        # Count occurrences of each label
        label_counts = Counter(label_ids)
        
        # Check for duplicates
        has_duplicates = any(count > 1 for count in label_counts.values())
        
        if has_duplicates:
            duplicate_count += 1
            max_count = max(label_counts.values())
            max_duplicates = max(max_duplicates, max_count)
            
            # Store first few examples for inspection
            if len(duplicate_examples) < 5:
                duplicate_examples.append({
                    'index': i,
                    'imageId': image_id,
                    'labelId': label_ids,
                    'counts': dict(label_counts)
                })
    
    # Print results
    print(f"Samples with duplicate labels: {duplicate_count}")
    print(f"Samples with empty labels: {empty_labels}")
    print(f"Maximum duplicate count: {max_duplicates}")
    print(f"Percentage with duplicates: {duplicate_count/len(annotations)*100:.2f}%")
    
    if duplicate_examples:
        print("\n" + "="*60)
        print("EXAMPLE DUPLICATES:")
        print("="*60)
        for example in duplicate_examples:
            print(f"Index: {example['index']}")
            print(f"Image ID: {example['imageId']}")
            print(f"Label IDs: {example['labelId']}")
            print(f"Label counts: {example['counts']}")
            print("-" * 40)
    
    # Test the one-hot encoding issue
    print("\n" + "="*60)
    print("ONE-HOT ENCODING TEST:")
    print("="*60)
    
    if duplicate_examples:
        test_example = duplicate_examples[0]
        labels = test_example['labelId']
        
        print(f"Testing with labels: {labels}")
        
        # Convert to tensor and apply one-hot
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        one_hot_sum = F.one_hot(labels_tensor, num_classes=131).sum(dim=0)
        
        print(f"One-hot sum unique values: {one_hot_sum.unique().tolist()}")
        print(f"Max value in one-hot sum: {one_hot_sum.max().item()}")
        print(f"Values > 1: {(one_hot_sum > 1).sum().item()} positions")
        
        # Show which classes have counts > 1
        problematic_classes = torch.where(one_hot_sum > 1)[0]
        if len(problematic_classes) > 0:
            print(f"Classes with duplicates: {problematic_classes.tolist()}")
            for cls in problematic_classes:
                print(f"  Class {cls.item()}: count = {one_hot_sum[cls].item()}")
    
    return {
        'total_samples': len(annotations),
        'duplicate_count': duplicate_count,
        'empty_labels': empty_labels,
        'max_duplicates': max_duplicates,
        'percentage_duplicates': duplicate_count/len(annotations)*100
    }

# Run the check
if __name__ == "__main__":
    results = check_validation_data("imat_data/val_annos_relabel.json")
    
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    for key, value in results.items():
        print(f"{key}: {value}")
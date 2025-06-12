import pandas as pd
import json
from pathlib import Path
from collections import defaultdict

def create_relabeling_script():
    """
    Create a relabeling script that takes train_annos_relabel-6-4.json (which already has labelId_new)
    and relabels it using the mapping from relabel-split.xlsx where labelId_new maps to group_label.
    The new labelId key maps to a dictionary containing each category and the corresponding group labels.
    """
    
    print("Loading relabel-split.xlsx mapping...")
    # Load the relabel mapping
    relabel_df = pd.read_excel('imat_data/relabel-split.xlsx', engine='openpyxl')
    
    # Create mapping dictionaries
    # Map labelId_new to group_label
    new_to_group = {}
    # Map labelId_new to category info (taskName)
    new_to_category = {}
    
    for _, row in relabel_df.iterrows():
        if pd.notna(row['labelId_new']) and pd.notna(row['group_label']):
            new_id = int(row['labelId_new'])
            group_id = int(row['group_label'])
            category = row['taskName']
            
            new_to_group[new_id] = group_id
            new_to_category[new_id] = category
    
    print(f"Created mappings for {len(new_to_group)} labels")
    print(f"Categories found: {set(new_to_category.values())}")
    print(f"Group labels range: {min(new_to_group.values())} to {max(new_to_group.values())}")
    
    print("\nLoading train_annos_relabel-6-4.json...")
    # Load the annotation data
    with open('imat_data/val_annos_clean.json', 'r') as f:
        data = json.load(f)
    
    print(f"Processing {len(data['annotations'])} annotations...")
    
    # Process annotations
    relabeled_annotations = []
    
    for i, annotation in enumerate(data['annotations']):
        if i % 10000 == 0:
            print(f"Processed {i}/{len(data['annotations'])} annotations...")
            
        image_id = annotation['imageId']
        new_labels = annotation['labelId']  # These are already labelId_new values
        
        # Create the new label structure
        new_annotation = {
            'imageId': image_id,
            'labelId': {},  # This will contain categories mapped to their group labels

        }
        
        # Group labels by category
        category_labels = defaultdict(list)
        category_group_labels = defaultdict(set)
        
        for new_label in new_labels:
            if new_label in new_to_group:
                group_label = new_to_group[new_label]
                category = new_to_category[new_label]
                

                category_group_labels[category].add(group_label)
        
        # Create the final labelId mapping: category -> list of group labels
        for category, group_set in category_group_labels.items():
            new_annotation['labelId'][category] = sorted(list(group_set))

        
        relabeled_annotations.append(new_annotation)
    
    # Create the new dataset
    relabeled_data = {
        'annotations': relabeled_annotations,
        'mapping_info': {
            'total_annotations': len(relabeled_annotations),
            'categories': list(set(new_to_category.values())),
            'total_group_labels': len(set(new_to_group.values())),
            'group_label_range': [min(new_to_group.values()), max(new_to_group.values())]
        }
    }
    
    # Save the relabeled dataset
    output_path = 'imat_data/val_annos_group_relabeled.json'
    print(f"\nSaving relabeled dataset to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(relabeled_data, f, indent=2)
    
    print(f"Successfully saved {len(relabeled_annotations)} relabeled annotations")
    
    # Print some statistics
    print("\n=== Relabeling Statistics ===")
    category_counts = defaultdict(int)
    group_label_counts = defaultdict(int)
    
    for annotation in relabeled_annotations:
        for category, group_labels in annotation['labelId'].items():
            category_counts[category] += 1
            for group_label in group_labels:
                group_label_counts[group_label] += 1
    
    print("Category distribution:")
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count} annotations")
    
    print("\nGroup label distribution (top 10):")
    for group_label, count in sorted(group_label_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  Group {group_label}: {count} occurrences")
    
    # Save a sample of the relabeled data for inspection
    sample_path = 'imat_data/train_annos_group_relabeled_sample.json'
    sample_data = {
        'annotations': relabeled_annotations[:100],  # First 100 annotations
        'mapping_info': relabeled_data['mapping_info']
    }
    
    with open(sample_path, 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    print(f"\nSample data (first 100 annotations) saved to {sample_path}")
    print("Done!")

if __name__ == "__main__":
    # Change to the correct directory
    import os
    os.chdir(Path(__file__).parent.parent)
    
    create_relabeling_script() 
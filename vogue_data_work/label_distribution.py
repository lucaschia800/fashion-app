import json
from collections import Counter

def analyze_label_distribution(file_path):
    """Analyze the distribution of labels in the fashion dataset"""
    
    print(f"Analyzing label distribution in: {file_path}")
    print("=" * 50)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Collect all labels
    all_labels = []
    
    for designer in data:
        if 'Shows' in designer and designer['Shows']:
            for show in designer['Shows']:
                if 'Looks' in show and show['Looks'] is not None:
                    for look in show['Looks']:
                        if look and 'Garments' in look:
                            garments = look['Garments']
                            if isinstance(garments, dict) and 'labels' in garments:
                                labels = garments['labels']
                                if isinstance(labels, list):
                                    all_labels.extend(labels)
    
    # Count label frequencies
    label_counts = Counter(all_labels)
    
    print(f"Total labels found: {len(all_labels)}")
    print(f"Unique labels: {len(label_counts)}")
    print(f"Label range: {min(label_counts.keys()) if label_counts else 'N/A'} to {max(label_counts.keys()) if label_counts else 'N/A'}")
    
    print(f"\nLabel distribution (top 20):")
    print("-" * 30)
    for label, count in label_counts.most_common(20):
        percentage = (count / len(all_labels)) * 100
        print(f"Label {label}: {count} times ({percentage:.1f}%)")
    
    # Show some statistics
    if label_counts:
        print(f"\nStatistics:")
        print(f"Most common label: {label_counts.most_common(1)[0]}")
        print(f"Least common label: {label_counts.most_common()[-1]}")
        print(f"Average frequency: {len(all_labels) / len(label_counts):.1f}")
        
        # Show labels that appear only once
        single_occurrence = [label for label, count in label_counts.items() if count == 1]
        if single_occurrence:
            print(f"Labels appearing only once: {len(single_occurrence)}")
            print(f"  Examples: {single_occurrence[:5]}")

if __name__ == "__main__":
    file_path = "fash-data/letter_A.json"
    analyze_label_distribution(file_path)
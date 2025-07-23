import json
from collections import defaultdict, Counter

def analyze_letter_A_data(json_path):
    """Analyze the letter_A.json file to understand data structure and patterns"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("=== LETTER A DATA ANALYSIS ===\n")
    
    # Basic structure analysis
    print(f"Number of designers: {len(data)}")
    
    total_shows = 0
    total_looks = 0
    total_with_urls = 0
    total_with_labels = 0
    total_without_labels = 0
    
    designer_stats = {}
    show_stats = defaultdict(int)
    
    for designer in data:
        designer_name = designer.get('Designer_Name', 'Unknown')
        shows = designer.get('Shows', [])
        
        designer_looks = 0
        designer_with_labels = 0
        designer_without_labels = 0
        
        for show in shows:
            if show is None:
                continue
                
            show_name = show.get('Show_Name', 'Unknown')
            looks = show.get('Looks', [])
            
            if looks is None:
                continue
                
            total_shows += 1
            show_stats[show_name] += len(looks)
            
            for look in looks:
                total_looks += 1
                designer_looks += 1
                
                url = look.get('Look_Url')
                if url:
                    total_with_urls += 1
                    
                    garments = look.get('Garments', {})
                    boxes = garments.get('boxes', [])
                    labels = garments.get('labels', [])
                    
                    if boxes and labels:
                        total_with_labels += 1
                        designer_with_labels += 1
                    else:
                        total_without_labels += 1
                        designer_without_labels += 1
        
        designer_stats[designer_name] = {
            'total_looks': designer_looks,
            'with_labels': designer_with_labels,
            'without_labels': designer_without_labels,
            'label_rate': designer_with_labels / max(designer_looks, 1) * 100
        }
    
    # Print overall statistics
    print(f"Total shows: {total_shows}")
    print(f"Total looks: {total_looks}")
    print(f"Looks with URLs: {total_with_urls}")
    print(f"Looks with labels: {total_with_labels}")
    print(f"Looks without labels: {total_without_labels}")
    print(f"Overall labeling rate: {total_with_labels/max(total_with_urls, 1)*100:.1f}%\n")
    
    # Designer analysis
    print("=== DESIGNER ANALYSIS ===")
    sorted_designers = sorted(designer_stats.items(), key=lambda x: x[1]['total_looks'], reverse=True)
    
    for designer, stats in sorted_designers[:10]:  # Top 10 designers
        print(f"{designer}:")
        print(f"  Total looks: {stats['total_looks']}")
        print(f"  With labels: {stats['with_labels']}")
        print(f"  Without labels: {stats['without_labels']}")
        print(f"  Label rate: {stats['label_rate']:.1f}%")
        print()
    
    # Show analysis
    print("=== SHOW ANALYSIS ===")
    sorted_shows = sorted(show_stats.items(), key=lambda x: x[1], reverse=True)
    
    print("Top 10 shows by number of looks:")
    for show, count in sorted_shows[:10]:
        print(f"  {show}: {count} looks")
    
    # Label distribution analysis
    print("\n=== LABEL DISTRIBUTION ===")
    all_labels = []
    for designer in data:
        for show in designer.get('Shows', []):
            if show is None:
                continue
            looks = show.get('Looks', [])
            if looks is None:
                continue
            for look in looks:
                garments = look.get('Garments', {})
                labels = garments.get('labels', [])
                if labels:
                    all_labels.extend(labels)
    
    if all_labels:
        label_counts = Counter(all_labels)
        print("Most common labels:")
        for label, count in label_counts.most_common(10):
            print(f"  {label}: {count} times")
    
    # Pattern analysis for unlabeled images
    print("\n=== UNLABELED IMAGE PATTERNS ===")
    unlabeled_designers = []
    for designer_name, stats in designer_stats.items():
        if stats['without_labels'] > 0:
            unlabeled_designers.append((designer_name, stats['without_labels']))
    
    unlabeled_designers.sort(key=lambda x: x[1], reverse=True)
    print("Designers with most unlabeled images:")
    for designer, count in unlabeled_designers[:10]:
        print(f"  {designer}: {count} unlabeled images")

if __name__ == "__main__":
    analyze_letter_A_data("fash-data/letter_A.json") 
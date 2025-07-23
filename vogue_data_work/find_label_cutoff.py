import json
from collections import defaultdict

def find_label_cutoff(json_path):
    """Find where labels stop appearing in the data, indicating potential rate limiting"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("=== LABEL CUTOFF ANALYSIS ===\n")
    
    # Track labels by position in the data
    label_positions = []
    no_label_positions = []
    
    total_processed = 0
    consecutive_no_labels = 0
    max_consecutive_no_labels = 0
    last_label_position = 0
    
    for designer_idx, designer in enumerate(data):
        designer_name = designer.get('Designer_Name', 'Unknown')
        shows = designer.get('Shows', [])
        
        for show_idx, show in enumerate(shows):
            if show is None:
                continue
                
            show_name = show.get('Show_Name', 'Unknown')
            looks = show.get('Looks', [])
            
            if looks is None:
                continue
                
            for look_idx, look in enumerate(looks):
                total_processed += 1
                
                url = look.get('Look_Url')
                if url:
                    garments = look.get('Garments', {})
                    boxes = garments.get('boxes', [])
                    labels = garments.get('labels', [])
                    
                    if boxes and labels:
                        label_positions.append(total_processed)
                        consecutive_no_labels = 0
                        last_label_position = total_processed
                    else:
                        no_label_positions.append(total_processed)
                        consecutive_no_labels += 1
                        max_consecutive_no_labels = max(max_consecutive_no_labels, consecutive_no_labels)
    
    print(f"Total images processed: {total_processed}")
    print(f"Images with labels: {len(label_positions)}")
    print(f"Images without labels: {len(no_label_positions)}")
    print(f"Last label found at position: {last_label_position}")
    print(f"Max consecutive images without labels: {max_consecutive_no_labels}")
    
    # Find the cutoff point
    if label_positions:
        print(f"\n=== CUTOFF ANALYSIS ===")
        print(f"First label at position: {min(label_positions)}")
        print(f"Last label at position: {max(label_positions)}")
        
        # Check if there's a clear cutoff
        if max_consecutive_no_labels > len(label_positions):
            print(f"\n🚨 SUSPICIOUS PATTERN DETECTED!")
            print(f"After position {last_label_position}, there are {total_processed - last_label_position} consecutive images with no labels")
            print(f"This suggests a rate limiting or timeout occurred around position {last_label_position}")
            
            # Show some context around the cutoff
            print(f"\n=== CONTEXT AROUND CUTOFF ===")
            cutoff_designer_idx = 0
            cutoff_show_idx = 0
            cutoff_look_idx = 0
            
            current_position = 0
            for designer_idx, designer in enumerate(data):
                shows = designer.get('Shows', [])
                for show_idx, show in enumerate(shows):
                    if show is None:
                        continue
                    looks = show.get('Looks', [])
                    if looks is None:
                        continue
                    for look_idx, look in enumerate(looks):
                        current_position += 1
                        if current_position == last_label_position:
                            cutoff_designer_idx = designer_idx
                            cutoff_show_idx = show_idx
                            cutoff_look_idx = look_idx
                            break
                    if current_position == last_label_position:
                        break
                if current_position == last_label_position:
                    break
            
            print(f"Last labeled image was in:")
            print(f"  Designer: {data[cutoff_designer_idx].get('Designer_Name', 'Unknown')}")
            print(f"  Show: {data[cutoff_designer_idx]['Shows'][cutoff_show_idx].get('Show_Name', 'Unknown')}")
            print(f"  Look index: {cutoff_look_idx}")
            
            # Show what comes after
            if cutoff_designer_idx < len(data) - 1:
                next_designer = data[cutoff_designer_idx + 1]
                print(f"\nNext designer after cutoff: {next_designer.get('Designer_Name', 'Unknown')}")
    
    # Analyze the distribution of labels
    if label_positions:
        print(f"\n=== LABEL DISTRIBUTION ANALYSIS ===")
        first_quarter = total_processed // 4
        second_quarter = total_processed // 2
        third_quarter = 3 * total_processed // 4
        
        labels_in_first_quarter = len([p for p in label_positions if p <= first_quarter])
        labels_in_second_quarter = len([p for p in label_positions if first_quarter < p <= second_quarter])
        labels_in_third_quarter = len([p for p in label_positions if second_quarter < p <= third_quarter])
        labels_in_fourth_quarter = len([p for p in label_positions if p > third_quarter])
        
        print(f"Labels in first quarter (1-{first_quarter}): {labels_in_first_quarter}")
        print(f"Labels in second quarter ({first_quarter+1}-{second_quarter}): {labels_in_second_quarter}")
        print(f"Labels in third quarter ({second_quarter+1}-{third_quarter}): {labels_in_third_quarter}")
        print(f"Labels in fourth quarter ({third_quarter+1}-{total_processed}): {labels_in_fourth_quarter}")

if __name__ == "__main__":
    find_label_cutoff("fash-data/letter_A.json") 
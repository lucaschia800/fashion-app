import json

def analyze_unknown_fields(json_path):
    """Analyze which fields show 'Unknown' and when they start appearing"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    print("=== UNKNOWN FIELD ANALYSIS ===\n")
    
    # Track when Unknown values start appearing
    first_unknown_designer = None
    first_unknown_show = None
    first_unknown_position = None
    
    # Track all Unknown occurrences
    unknown_designers = []
    unknown_shows = []
    
    total_processed = 0
    
    for designer_idx, designer in enumerate(data):
        designer_name = designer.get('Designer_Name', 'Unknown')
        shows = designer.get('Shows', [])
        
        # Check if this is the first Unknown designer
        if designer_name == 'Unknown' and first_unknown_designer is None:
            first_unknown_designer = designer_idx
            first_unknown_position = total_processed
            print(f"🚨 FIRST 'Unknown' designer found at position {total_processed} (designer index {designer_idx})")
        
        if designer_name == 'Unknown':
            unknown_designers.append(designer_idx)
        
        for show_idx, show in enumerate(shows):
            if show is None:
                continue
                
            show_name = show.get('Show_Name', 'Unknown')
            looks = show.get('Looks', [])
            
            if looks is None:
                continue
                
            # Check if this is the first Unknown show
            if show_name == 'Unknown' and first_unknown_show is None:
                first_unknown_show = (designer_idx, show_idx)
                print(f"🚨 FIRST 'Unknown' show found at designer {designer_idx}, show {show_idx}")
            
            if show_name == 'Unknown':
                unknown_shows.append((designer_idx, show_idx))
            
            for look_idx, look in enumerate(looks):
                total_processed += 1
                
                # Check for other potential Unknown fields in look data
                url = look.get('Look_Url')
                if url == 'Unknown':
                    print(f"🚨 'Unknown' URL found at position {total_processed}")
                
                garments = look.get('Garments', {})
                if garments == 'Unknown':
                    print(f"🚨 'Unknown' Garments found at position {total_processed}")
    
    print(f"\n=== SUMMARY ===")
    print(f"Total designers: {len(data)}")
    print(f"Designers with 'Unknown' name: {len(unknown_designers)}")
    print(f"Shows with 'Unknown' name: {len(unknown_shows)}")
    print(f"Total looks processed: {total_processed}")
    
    if first_unknown_designer is not None:
        print(f"\n=== FIRST UNKNOWN OCCURRENCE ===")
        print(f"First 'Unknown' designer at index: {first_unknown_designer}")
        print(f"Position in data: {first_unknown_position}")
        
        # Show context around the first Unknown
        if first_unknown_designer > 0:
            prev_designer = data[first_unknown_designer - 1]
            print(f"\nPrevious designer (last known): {prev_designer.get('Designer_Name', 'Unknown')}")
            print(f"Number of shows for previous designer: {len(prev_designer.get('Shows', []))}")
        
        current_designer = data[first_unknown_designer]
        print(f"\nFirst 'Unknown' designer details:")
        print(f"  Designer_Name field: {current_designer.get('Designer_Name', 'MISSING')}")
        print(f"  Number of shows: {len(current_designer.get('Shows', []))}")
        
        # Check if the field is actually missing or just set to "Unknown"
        if 'Designer_Name' not in current_designer:
            print(f"  Designer_Name field is MISSING from JSON")
        elif current_designer['Designer_Name'] is None:
            print(f"  Designer_Name field is NULL")
        elif current_designer['Designer_Name'] == '':
            print(f"  Designer_Name field is empty string")
        else:
            print(f"  Designer_Name field contains: '{current_designer['Designer_Name']}'")
    
    # Analyze the pattern of Unknown values
    if unknown_designers:
        print(f"\n=== UNKNOWN PATTERN ANALYSIS ===")
        print(f"Unknown designers appear at indices: {unknown_designers[:10]}...")  # First 10
        
        # Check if Unknown becomes consistent after a point
        if len(unknown_designers) > 1:
            first_unknown_idx = unknown_designers[0]
            consecutive_unknown = 0
            for i in range(first_unknown_idx, len(data)):
                if data[i].get('Designer_Name', 'Unknown') == 'Unknown':
                    consecutive_unknown += 1
                else:
                    break
            
            print(f"After first 'Unknown' designer, {consecutive_unknown} consecutive designers are 'Unknown'")
            if consecutive_unknown > len(data) * 0.5:
                print(f"🚨 This suggests a systematic failure - most designers after position {first_unknown_idx} are 'Unknown'")

if __name__ == "__main__":
    analyze_unknown_fields("fash-data/letter_A.json") 
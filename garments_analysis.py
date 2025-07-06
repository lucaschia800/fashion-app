import json

def analyze_garments_structure(file_path):
    """Analyze the Garments structure in detail"""
    
    print(f"Analyzing Garments structure in: {file_path}")
    print("=" * 60)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Total designers: {len(data)}")
    
    # Find items with non-empty Garments
    items_with_garments = []
    empty_garments_count = 0
    none_looks_count = 0
    
    for i, designer in enumerate(data):
        if 'Shows' in designer and designer['Shows']:
            for show in designer['Shows']:
                if 'Looks' in show and show['Looks'] is not None:
                    for look in show['Looks']:
                        if look and 'Garments' in look:
                            garments = look['Garments']
                            if isinstance(garments, dict) and garments:  # Non-empty dict
                                items_with_garments.append({
                                    'designer_idx': i,
                                    'designer': designer.get('Designer', 'Unknown'),
                                    'show_name': show.get('Show Name', 'Unknown'),
                                    'look_number': look.get('Look_Number', 'Unknown'),
                                    'garments': garments
                                })
                            elif isinstance(garments, dict) and not garments:  # Empty dict
                                empty_garments_count += 1
                elif 'Looks' in show and show['Looks'] is None:
                    none_looks_count += 1
    
    print(f"Shows with None Looks: {none_looks_count}")
    print(f"Looks with empty Garments: {empty_garments_count}")
    print(f"Looks with non-empty Garments: {len(items_with_garments)}")
    
    if items_with_garments:
        print(f"\n{'='*60}")
        print("SAMPLE ITEMS WITH GARMENTS DATA")
        print(f"{'='*60}")
        
        # Show first few items with garments
        for i, item in enumerate(items_with_garments[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Designer: {item['designer']}")
            print(f"Show: {item['show_name']}")
            print(f"Look Number: {item['look_number']}")
            print(f"Garments keys: {list(item['garments'].keys())}")
            
            # Show detailed garments structure
            for key, value in item['garments'].items():
                if isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                    if len(value) > 0:
                        print(f"    First item type: {type(value[0])}")
                        if isinstance(value[0], list):
                            print(f"    First item length: {len(value[0])}")
                            print(f"    Sample: {value[0]}")
                        else:
                            print(f"    Sample: {value[0]}")
                else:
                    print(f"  {key}: {type(value).__name__} = {value}")
    
    # Analyze the structure pattern
    print(f"\n{'='*60}")
    print("GARMENTS STRUCTURE PATTERN")
    print(f"{'='*60}")
    
    if items_with_garments:
        first_garments = items_with_garments[0]['garments']
        print("Typical Garments structure:")
        for key, value in first_garments.items():
            if isinstance(value, list):
                print(f"  {key}: list of {len(value)} items")
                if len(value) > 0:
                    if isinstance(value[0], list):
                        print(f"    Each item is a list of {len(value[0])} numbers")
                        print(f"    Sample item: {value[0]}")
                    else:
                        print(f"    Each item is a {type(value[0]).__name__}")
                        print(f"    Sample item: {value[0]}")
            else:
                print(f"  {key}: {type(value).__name__} = {value}")
    
    # Show distribution of garment counts
    print(f"\n{'='*60}")
    print("GARMENT COUNT DISTRIBUTION")
    print(f"{'='*60}")
    
    if items_with_garments:
        garment_counts = []
        for item in items_with_garments:
            if 'boxes' in item['garments']:
                garment_counts.append(len(item['garments']['boxes']))
        
        if garment_counts:
            print(f"Number of garments per look (boxes):")
            print(f"  Min: {min(garment_counts)}")
            print(f"  Max: {max(garment_counts)}")
            print(f"  Average: {sum(garment_counts)/len(garment_counts):.1f}")
            
            # Show some examples
            print(f"\nExamples of looks with different garment counts:")
            for count in sorted(set(garment_counts))[:5]:  # Show first 5 unique counts
                examples = [item for item in items_with_garments if len(item['garments'].get('boxes', [])) == count]
                print(f"  {count} garments: {len(examples)} looks")
                if examples:
                    sample = examples[0]
                    print(f"    Example: {sample['designer']} - {sample['show_name']} Look {sample['look_number']}")

if __name__ == "__main__":
    file_path = "fash-data/letter_A.json"
    analyze_garments_structure(file_path) 
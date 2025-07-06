import json
from collections import defaultdict

def analyze_json_detailed(file_path):
    """Detailed analysis of JSON structure showing nesting and patterns"""
    
    print(f"Analyzing: {file_path}")
    print("=" * 60)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Root type: {type(data)}")
    
    if isinstance(data, list):
        print(f"Number of items: {len(data)}")
        if len(data) > 0:
            print(f"\n{'='*60}")
            print("DETAILED STRUCTURE ANALYSIS")
            print(f"{'='*60}")
            
            # Analyze first item in detail
            print(f"\n1. FIRST ITEM STRUCTURE:")
            analyze_item_recursive(data[0], prefix="  ", max_depth=5)
            
            # Compare first few items to find patterns
            print(f"\n{'='*60}")
            print("2. PATTERN ANALYSIS (comparing first 3 items)")
            print(f"{'='*60}")
            
            for i, item in enumerate(data[:3]):
                print(f"\n--- Item {i+1} ---")
                analyze_item_structure(item, prefix="  ")
            
            # Find common patterns
            print(f"\n{'='*60}")
            print("3. COMMON PATTERNS")
            print(f"{'='*60}")
            find_common_patterns(data[:5])  # Analyze first 5 items
            
            # Show sample data
            print(f"\n{'='*60}")
            print("4. SAMPLE DATA VALUES")
            print(f"{'='*60}")
            show_sample_values(data[0])

def analyze_item_recursive(item, prefix="", max_depth=5, current_depth=0):
    """Recursively analyze an item's structure"""
    if current_depth >= max_depth:
        print(f"{prefix}... (max depth reached)")
        return
    
    if isinstance(item, dict):
        print(f"{prefix}Dict with {len(item)} keys:")
        for key, value in item.items():
            if isinstance(value, list):
                print(f"{prefix}  {key}: list with {len(value)} items")
                if len(value) > 0 and current_depth < max_depth - 1:
                    print(f"{prefix}    First item:")
                    analyze_item_recursive(value[0], prefix + "      ", max_depth, current_depth + 1)
            elif isinstance(value, dict):
                print(f"{prefix}  {key}: dict with {len(value)} keys")
                if current_depth < max_depth - 1:
                    analyze_item_recursive(value, prefix + "    ", max_depth, current_depth + 1)
            else:
                print(f"{prefix}  {key}: {type(value).__name__} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
    
    elif isinstance(item, list):
        print(f"{prefix}List with {len(item)} items")
        if len(item) > 0 and current_depth < max_depth - 1:
            print(f"{prefix}  First item:")
            analyze_item_recursive(item[0], prefix + "    ", max_depth, current_depth + 1)
    else:
        print(f"{prefix}{type(item).__name__} = {str(item)[:100]}{'...' if len(str(item)) > 100 else ''}")

def analyze_item_structure(item, prefix=""):
    """Analyze structure without going too deep"""
    if isinstance(item, dict):
        for key, value in item.items():
            if isinstance(value, list):
                print(f"{prefix}{key}: list with {len(value)} items")
                if len(value) > 0:
                    if isinstance(value[0], dict):
                        print(f"{prefix}  First item keys: {list(value[0].keys())}")
                    else:
                        print(f"{prefix}  First item type: {type(value[0])}")
            elif isinstance(value, dict):
                print(f"{prefix}{key}: dict with keys: {list(value.keys())}")
            else:
                print(f"{prefix}{key}: {type(value).__name__}")

def find_common_patterns(items):
    """Find common patterns across items"""
    if not items:
        return
    
    # Analyze structure consistency
    print("Structure consistency across items:")
    
    # Check if all items have the same top-level keys
    first_keys = set(items[0].keys()) if isinstance(items[0], dict) else set()
    all_same_keys = all(set(item.keys()) == first_keys for item in items if isinstance(item, dict))
    
    print(f"  All items have same top-level keys: {all_same_keys}")
    if all_same_keys:
        print(f"  Top-level keys: {list(first_keys)}")
    
    # Analyze Shows structure
    if 'Shows' in first_keys:
        print(f"\nShows analysis:")
        show_counts = [len(item['Shows']) for item in items if isinstance(item, dict) and 'Shows' in item]
        print(f"  Number of shows per designer: {show_counts}")
        
        # Check if all shows have same structure
        if items[0]['Shows']:
            first_show_keys = set(items[0]['Shows'][0].keys())
            print(f"  First show keys: {list(first_show_keys)}")
            
            # Check Looks structure
            if 'Looks' in first_show_keys:
                look_counts = [len(show['Looks']) for item in items[:3] for show in item['Shows'] if 'Looks' in show]
                print(f"  Number of looks per show (first 3 designers): {look_counts}")
                
                if items[0]['Shows'][0]['Looks']:
                    first_look_keys = set(items[0]['Shows'][0]['Looks'][0].keys())
                    print(f"  First look keys: {list(first_look_keys)}")

def show_sample_values(item):
    """Show actual sample values from the data"""
    if isinstance(item, dict):
        print("Sample values from first item:")
        for key, value in item.items():
            if isinstance(value, str):
                print(f"  {key}: {value}")
            elif isinstance(value, list):
                print(f"  {key}: list with {len(value)} items")
                if len(value) > 0:
                    if isinstance(value[0], dict):
                        print(f"    First item keys: {list(value[0].keys())}")
                        # Show a sample from the first nested item
                        if 'Show Name' in value[0]:
                            print(f"    Sample show name: {value[0]['Show Name']}")
                        if 'Looks' in value[0] and value[0]['Looks']:
                            first_look = value[0]['Looks'][0]
                            print(f"    Sample look keys: {list(first_look.keys())}")
                            if 'Look_Url' in first_look:
                                print(f"    Sample look URL: {first_look['Look_Url'][:80]}...")
                            if 'Garments' in first_look:
                                garments = first_look['Garments']
                                if isinstance(garments, dict):
                                    print(f"    Garments keys: {list(garments.keys())}")
                                    if 'boxes' in garments:
                                        print(f"    Number of garment boxes: {len(garments['boxes'])}")
                                        if garments['boxes']:
                                            print(f"    Sample box: {garments['boxes'][0]}")

if __name__ == "__main__":
    file_path = "fash-data/letter_A.json"
    analyze_json_detailed(file_path) 
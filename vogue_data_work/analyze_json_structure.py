import json
from collections import defaultdict
import os

def analyze_json_structure(file_path):
    """Analyze the structure of a JSON file and print a summary"""
    
    print(f"Analyzing: {file_path}")
    print("=" * 50)
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    print(f"Type: {type(data)}")
    
    if isinstance(data, list):
        print(f"Length: {len(data)} items")
        if len(data) > 0:
            print(f"First item type: {type(data[0])}")
            print(f"First item keys: {list(data[0].keys()) if isinstance(data[0], dict) else 'Not a dict'}")
            
            # Show structure of first few items
            for i, item in enumerate(data[:3]):
                print(f"\n--- Item {i+1} ---")
                analyze_item(item, indent=2)
                
    elif isinstance(data, dict):
        print(f"Keys: {list(data.keys())}")
        for key, value in data.items():
            print(f"\n--- Key: {key} ---")
            analyze_item(value, indent=2)

def analyze_item(item, indent=0):
    """Recursively analyze an item's structure"""
    spaces = " " * indent
    
    if isinstance(item, dict):
        print(f"{spaces}Type: dict with {len(item)} keys")
        print(f"{spaces}Keys: {list(item.keys())}")
        
        # Show sample values for each key
        for key, value in item.items():
            if isinstance(value, list):
                print(f"{spaces}  {key}: list with {len(value)} items")
                if len(value) > 0:
                    print(f"{spaces}    First item type: {type(value[0])}")
                    if isinstance(value[0], dict):
                        print(f"{spaces}    First item keys: {list(value[0].keys())}")
            elif isinstance(value, dict):
                print(f"{spaces}  {key}: dict with {len(value)} keys")
                print(f"{spaces}    Keys: {list(value.keys())}")
            else:
                print(f"{spaces}  {key}: {type(value).__name__} = {str(value)[:100]}{'...' if len(str(value)) > 100 else ''}")
                
    elif isinstance(item, list):
        print(f"{spaces}Type: list with {len(item)} items")
        if len(item) > 0:
            print(f"{spaces}First item type: {type(item[0])}")
            if isinstance(item[0], dict):
                print(f"{spaces}First item keys: {list(item[0].keys())}")
    else:
        print(f"{spaces}Type: {type(item).__name__} = {str(item)[:100]}{'...' if len(str(item)) > 100 else ''}")

def count_items_by_type(data, path=""):
    """Count different types of items in the JSON"""
    counts = defaultdict(int)
    
    if isinstance(data, dict):
        counts[f"{path}dict"] += 1
        for key, value in data.items():
            new_path = f"{path}.{key}" if path else key
            sub_counts = count_items_by_type(value, new_path)
            for k, v in sub_counts.items():
                counts[k] += v
                
    elif isinstance(data, list):
        counts[f"{path}list"] += 1
        for i, item in enumerate(data):
            new_path = f"{path}[{i}]"
            sub_counts = count_items_by_type(item, new_path)
            for k, v in sub_counts.items():
                counts[k] += v
    else:
        counts[f"{path}{type(data).__name__}"] += 1
    
    return counts

if __name__ == "__main__":
    file_path = "fash-data/letter_A.json"
    
    if os.path.exists(file_path):
        analyze_json_structure(file_path)
        
        # Also show some statistics
        print("\n" + "=" * 50)
        print("DETAILED STATISTICS")
        print("=" * 50)
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        counts = count_items_by_type(data)
        print("\nItem counts by type:")
        for item_type, count in sorted(counts.items()):
            print(f"  {item_type}: {count}")
            
    else:
        print(f"File not found: {file_path}") 
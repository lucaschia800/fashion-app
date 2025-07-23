import json

def examine_first_100(json_path, output_file="first_100_entries.txt"):
    """Output the first 100 entries in a readable format to a file"""
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    with open(output_file, 'w') as out_f:
        out_f.write("=== FIRST 100 ENTRIES EXAMINATION ===\n\n")
        
        for i, designer in enumerate(data[:100]):
            out_f.write(f"ENTRY {i+1}:\n")
            out_f.write(f"  Designer_Name: {designer.get('Designer_Name', 'MISSING')}\n")
            
            shows = designer.get('Shows', [])
            out_f.write(f"  Number of shows: {len(shows)}\n")
            
            for j, show in enumerate(shows):
                if show is None:
                    out_f.write(f"    Show {j+1}: NULL\n")
                    continue
                    
                show_name = show.get('Show_Name', 'MISSING')
                out_f.write(f"    Show {j+1}: {show_name}\n")
                
                looks = show.get('Looks', [])
                if looks is None:
                    out_f.write(f"      Looks: NULL\n")
                    continue
                    
                out_f.write(f"      Number of looks: {len(looks)}\n")
                
                # Show first few looks for each show
                for k, look in enumerate(looks[:3]):  # Only first 3 looks per show
                    url = look.get('Look_Url', 'MISSING')
                    garments = look.get('Garments', {})
                    look_number = look.get('Look_Number', 'MISSING')
                    
                    out_f.write(f"        Look {k+1}:\n")
                    out_f.write(f"          URL: {url}\n")
                    out_f.write(f"          Look_Number: {look_number}\n")
                    out_f.write(f"          Garments keys: {list(garments.keys())}\n")
                    
                    if 'boxes' in garments and 'labels' in garments:
                        boxes = garments['boxes']
                        labels = garments['labels']
                        out_f.write(f"          Boxes: {len(boxes) if boxes else 0}\n")
                        out_f.write(f"          Labels: {labels if labels else 'None'}\n")
                    else:
                        out_f.write(f"          Garments structure: {garments}\n")
                
                if len(looks) > 3:
                    out_f.write(f"        ... and {len(looks) - 3} more looks\n")
            
            out_f.write("-" * 50 + "\n")
            
            # Stop after 100 entries
            if i == 99:
                break
    
    print(f"Output saved to: {output_file}")

if __name__ == "__main__":
    examine_first_100("fash-data/letter_A.json") 
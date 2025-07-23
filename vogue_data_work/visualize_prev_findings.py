import json
import requests
import random
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

def plot_bboxes_from_json(json_path, n_per_page=8):
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Reservoir sampling to get random samples without collecting all first
    samples = []
    count = 0
    stats = {'with_labels': 0, 'without_labels': 0, 'total_processed': 0}
    
    for designer in data:
        designer_name = designer.get('Designer', 'Unknown')
        
        # Only process designers whose name starts with 'A' (case-insensitive)
        if not designer_name.lower().startswith('a'):
            continue
            
        for show in designer.get('Shows', []):
            if show is None:  # Skip if show is None
                continue
            looks = show.get('Looks', [])
            if looks is None:  # Skip if Looks is None
                continue
            for look in looks:
                url = look.get('Look_Url')
                if url:  # Only requirement is having a URL
                    count += 1
                    stats['total_processed'] += 1
                    garments = look.get('Garments', {})
                    boxes = garments.get('boxes', [])
                    labels = garments.get('labels', [])
                    
                    # Track statistics
                    if boxes and labels:
                        stats['with_labels'] += 1
                    else:
                        stats['without_labels'] += 1
                    
                    # Reservoir sampling: replace with probability k/n
                    if len(samples) < n_per_page:
                        samples.append((url, boxes, labels, designer.get('Designer_Name', 'Unknown'), show.get('Show_Name', 'Unknown')))
                    else:
                        # Randomly replace an existing sample
                        if random.random() < n_per_page / count:
                            replace_idx = random.randint(0, n_per_page - 1)
                            samples[replace_idx] = (url, boxes, labels, designer.get('Designer_Name', 'Unknown'), show.get('Show_Name', 'Unknown'))

    # Print statistics
    print(f"Total images processed: {stats['total_processed']}")
    if stats['total_processed'] > 0:
        print(f"Images with labels: {stats['with_labels']} ({stats['with_labels']/stats['total_processed']*100:.1f}%)")
        print(f"Images without labels: {stats['without_labels']} ({stats['without_labels']/stats['total_processed']*100:.1f}%)")
    else:
        print("No images found for designers starting with 'A'")
    print(f"Random sample of {len(samples)} images:")

    # Display the randomly sampled images
    if samples:
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        for j, ax in enumerate(axes.flatten()):
            if j >= len(samples):
                ax.axis('off')
                continue
            url, boxes, labels, designer, show = samples[j]
            try:
                response = requests.get(url, timeout=10)
                img = Image.open(BytesIO(response.content)).convert("RGB")
                draw = ImageDraw.Draw(img)
                
                # Only draw boxes and labels if they exist
                if boxes and labels:
                    for box, label in zip(boxes, labels):
                        box = [int(coord) for coord in box]
                        draw.rectangle(box, outline='red', width=3)
                        
                        # Try to use a larger font, fallback to default if not available
                        try:
                            # Try to use a larger font size
                            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", 24)
                        except:
                            try:
                                # Fallback to a different font
                                font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 24)
                            except:
                                # Use default font with larger size
                                font = ImageFont.load_default()
                        
                        # Draw text with black outline for better visibility
                        text = str(label)
                        text_bbox = draw.textbbox((box[0], box[1]), text, font=font)
                        
                        # Draw black outline
                        for dx in [-2, -1, 0, 1, 2]:
                            for dy in [-2, -1, 0, 1, 2]:
                                if dx != 0 or dy != 0:
                                    draw.text((box[0] + dx, box[1] + dy), text, fill='black', font=font)
                        
                        # Draw main text
                        draw.text((box[0], box[1]), text, fill='yellow', font=font)
                    
                    title = f"Labels: {labels}\n{designer}"
                else:
                    title = f"No labels detected\n{designer}"
                
                ax.imshow(img)
                ax.set_title(title, fontsize=8)
                ax.axis('off')
            except Exception as e:
                ax.set_title(f"Failed to load\n{designer}", fontsize=8)
                ax.axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No images found with URLs")

if __name__ == "__main__":
    plot_bboxes_from_json("fash-data/letter_A.json")
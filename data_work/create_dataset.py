import json
import os
import requests
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
import time

def find_last_downloaded_index(images, output_dir):
    """Find the index of the last successfully downloaded image in sequence."""
    if not output_dir.exists():
        return 0
    
    existing_image_ids = set()
    for file_path in output_dir.glob("*"):
        if file_path.is_file() and not file_path.name.endswith('.json'):
            image_id = file_path.stem
            existing_image_ids.add(image_id)
    
    if not existing_image_ids:
        return 0
    
    # Find the highest imageId that was downloaded
    max_downloaded_id = max(existing_image_ids, key=lambda x: int(x) if x.isdigit() else 0)
    
    # Find its position in the images list
    for i, img in enumerate(images):
        if str(img['imageId']) == max_downloaded_id:
            return i + 1  # Start from next position
    
    return 0

def download_image(url, image_id, output_dir):
    """Download a single image and save it with the imageId as filename."""
    try:
        # Get file extension from URL or default to .jpg
        parsed_url = urlparse(url)
        file_ext = os.path.splitext(parsed_url.path)[1]
        if not file_ext:
            file_ext = '.jpg'
        
        filename = f"{image_id}{file_ext}"
        filepath = output_dir / filename
        
        # Skip if file already exists
        if filepath.exists():
            print(f"Skipping {filename} - already exists")
            return True, filename, "Already exists"
        
        # Download the image
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Save the image
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Downloaded: {filename}")
        return True, filename, "Success"
        
    except Exception as e:
        print(f"Failed to download image {image_id}: {str(e)}")
        return False, image_id, str(e)

def create_dataset(json_file_path, max_workers=7):
    """
    Create image dataset from JSON file.
    
    Args:
        json_file_path (str): Path to JSON file containing image data
        max_workers (int): Number of concurrent downloads (default: 5)
    """
    
    # Create output directory
    output_dir = Path("imat_data/img")
    output_dir.mkdir(exist_ok=True)
    
    # Load JSON data
    try:
        with open(json_file_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file '{json_file_path}' not found")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in file '{json_file_path}'")
        return
    
    images = data.get('images', [])
    if not images:
        print("No images found in JSON data")
        return
    
    print(f"Total images in dataset: {len(images)}")
    print(f"Output directory: {output_dir.absolute()}")
    
    # Find resume point
    resume_index = find_last_downloaded_index(images, output_dir)
    print(f"Resuming from index: {resume_index}")
    print(f"Already downloaded: {resume_index} images")
    print(f"Remaining: {len(images) - resume_index} images")
    
    if resume_index >= len(images):
        print("All images already downloaded!")
        return
    
    # Get images to download (from resume point)
    images_to_download = images[resume_index:]
    
    # Download images concurrently
    successful_downloads = 0
    failed_downloads = []
    
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all download tasks
        future_to_image = {
            executor.submit(download_image, img['url'], img['imageId'], output_dir): img
            for img in images_to_download
        }
        
        # Process completed downloads
        for future in as_completed(future_to_image):
            success, identifier, message = future.result()
            if success:
                successful_downloads += 1
            else:
                failed_downloads.append((identifier, message))
    
    # Print summary
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n--- Download Summary ---")
    print(f"Images processed this session: {len(images_to_download)}")
    print(f"Successful downloads: {successful_downloads}")
    print(f"Failed downloads: {len(failed_downloads)}")
    print(f"Time elapsed: {elapsed_time:.2f} seconds")
    print(f"Total completion: {(resume_index + successful_downloads)}/{len(images)} ({((resume_index + successful_downloads)/len(images))*100:.1f}%)")
    
    if failed_downloads:
        print(f"\nFailed downloads:")
        for image_id, error in failed_downloads[:10]:
            print(f"  Image {image_id}: {error}")
        if len(failed_downloads) > 10:
            print(f"  ... and {len(failed_downloads) - 10} more")
    
    # Create a summary file
    summary = {
        "total_images": len(images),
        "resume_index": resume_index,
        "successful_downloads": successful_downloads,
        "failed_downloads": len(failed_downloads),
        "failed_list": failed_downloads,
        "download_time_seconds": elapsed_time,
        "completion_percentage": ((resume_index + successful_downloads)/len(images))*100
    }
    
    with open(output_dir / "download_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDataset updated in '{output_dir}' folder")
    print(f"Summary saved to '{output_dir}/download_summary.json'")

if __name__ == "__main__":
    # Usage example
    json_file = "imat_data/train.json"  # Replace with your JSON file path
    
    # Create the dataset
    create_dataset(json_file, max_workers=7)
    
    # Alternative: If you have the JSON data as a string or dict
    # You can also use it directly like this:
    """
    json_data = {
        "images": [
            {
                "url": "https://contestimg.wish.com/api/webimage/568e16a72dfd0133cb3f7a79-large", 
                "imageId": "1"
            },
            # ... more images
        ]
    }
    
    # Save to temporary file and process
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(json_data, f)
        temp_file = f.name
    
    create_dataset(temp_file)
    os.unlink(temp_file)  # Clean up temp file
    """
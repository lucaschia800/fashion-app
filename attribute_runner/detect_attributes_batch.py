import json
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import copy

import torch
from PIL import Image
import requests
from tqdm import tqdm
from torchvision.models import EfficientNet_V2_M_Weights

import classes.multiheaded_fefficient_v2 as mhf

# ------------------------------------------------------------
# Model & preprocessing helper
# ------------------------------------------------------------

def get_model(ckpt_path: str, device: str = "cuda"):
    """Load the multi-head EfficientNet model and its default transforms."""
    model = mhf.MultiHead_FEfficientNet(ckpt_path=ckpt_path).to(device)
    model.eval()
    transforms = EfficientNet_V2_M_Weights.DEFAULT.transforms()
    return model, transforms

# ------------------------------------------------------------
# Concurrent image download & cropping
# ------------------------------------------------------------

def _fetch_and_crop(task: Tuple[str, List[List[int]], List[int], Dict, torch.nn.Module]):
    """Download one image, crop every garment box, return list of (tensor, meta)."""
    url, boxes, labels, meta, tfm = task
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        print(f"⚠️  {url} → {exc}")
        return []

    out = []
    for idx, ((x1, y1, x2, y2), label) in enumerate(zip(boxes, labels)):
        crop = img.crop((x1, y1, x2, y2))
        out.append((tfm(crop), {**meta, "box_idx": idx, "label": label}))
    return out

# ------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------

def _infer_batch(model, img_list: List[torch.Tensor], meta_list: List[Dict], device: str):
    """Run one batch through the network and update the dataset with predictions."""
    batch = torch.stack(img_list).to(device)
    categories = {'gender': 3, 'material': 23, 'pattern': 18, 'style': 10, 'sleeve': 4, 'category': 48, 'color': 19}
    multiclass_categories = ['gender', 'style', 'sleeve']
    multilabel_categories = ['color', 'pattern', 'material', 'category']

    with torch.no_grad():
        logits = model(batch)  # dict[attr] -> Tensor[B, C]

    # Process predictions for each item in the batch
    for i, meta in enumerate(meta_list):
        # Direct access to garment dictionary using stored reference
        garment_dict = meta["garment_ref"]
        box_idx = meta["box_idx"]
        label = meta["label"]
        
        # Initialize attributes if not present
        if "attributes" not in garment_dict:
            num_boxes = len(garment_dict["boxes"])
            garment_dict["attributes"] = [{} for _ in range(num_boxes)]

        # Initialize this box's attributes if not present
        if box_idx >= len(garment_dict["attributes"]):
            # Extend attributes list if needed
            while len(garment_dict["attributes"]) <= box_idx:
                garment_dict["attributes"].append({})

        # Process predictions for each category
        for category in categories:
            if label == 9:  # Skip processing for label 9, store empty
                garment_dict["attributes"][box_idx][category] = []
            elif category in multiclass_categories:
                # Take top prediction for multiclass
                probabilities = torch.softmax(logits[category], dim=1)
                predicted_idx = torch.argmax(probabilities[i]).item()
                garment_dict["attributes"][box_idx][category] = [predicted_idx]
            else:  # multilabel categories
                # Take predictions above threshold
                probabilities = torch.sigmoid(logits[category])
                above_threshold = probabilities[i] > 0.5
                predicted_indices = torch.where(above_threshold)[0].tolist()
                garment_dict["attributes"][box_idx][category] = predicted_indices

        print(f"Updated {meta['designer']} | look {meta['look_id']} | box {meta['box_idx']} | label {label}")

# ------------------------------------------------------------
# Main detection routine
# ------------------------------------------------------------

def detect_attributes(
    model,
    dataset: List[Dict],
    letter: str,
    batch_size: int = 16,
    device: str = "cuda",
    transforms=None,
    max_workers: int = 16,
):
    """Download images concurrently, crop boxes, and perform batched inference."""

    batch_imgs: List[torch.Tensor] = []
    batch_meta: List[Dict] = []
    tasks = []

    # Build task list with direct references
    for designer in dataset:
        if not designer["Designer_Name"].startswith(letter):
            continue
        for show in designer["Shows"]:
            for look in show["Looks"]:
                tasks.append(
                    (
                        look["Look_Url"],
                        look["Garments"]["boxes"],
                        look["Garments"]["labels"],
                        {
                            "designer": designer["Designer_Name"],
                            "show_id": show.get("Show_Name", "N/A"),
                            "look_id": look.get("Look_Number", "N/A"),
                            "garment_ref": look["Garments"]  # Direct reference!
                        },
                        transforms,
                    )
                )

    # Concurrent fetching & cropping
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for future in tqdm(as_completed([pool.submit(_fetch_and_crop, t) for t in tasks]), total=len(tasks), desc="Fetching"):
            for tensor, meta in future.result():
                batch_imgs.append(tensor)
                batch_meta.append(meta)
                if len(batch_imgs) == batch_size:
                    _infer_batch(model, batch_imgs, batch_meta, device)
                    batch_imgs.clear()
                    batch_meta.clear()

    # flush tail
    if batch_imgs:
        _infer_batch(model, batch_imgs, batch_meta, device)

# ------------------------------------------------------------
# Entry point
# ------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tfm = get_model("weights/Mefficientnet_v2_freeze_ckpt2.pth", device)

    with open("fash-data/letter_A.json", "r") as f:
        dataset = json.load(f)
    
    # Make a deep copy to avoid modifying the original
    dataset_with_attributes = copy.deepcopy(dataset)
    
    detect_attributes(
        model,
        dataset_with_attributes,
        letter="A",
        batch_size=100,
        device=device,
        transforms=tfm,
        max_workers=16,
    )
    
    # Save the updated dataset
    with open("fash-data/letter_A_with_attributes.json", "w") as f:
        json.dump(dataset_with_attributes, f, indent=2)
    
    print("Saved updated dataset to letter_A_with_attributes.json")


if __name__ == "__main__":
    main() 

import json
import copy
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional

import requests
from PIL import Image
from tqdm import tqdm

import torch
from torchvision.transforms import v2 as transforms
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ------------------------------------------------------------
# Model & preprocessing helpers
# ------------------------------------------------------------

def get_model(num_classes: int, weights_path: str, device: str):
    """Load Faster R-CNN, swap the predictor for num_classes, load ckpt."""
    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT,
        box_detections_per_img=8,
        box_score_thresh=0.83,
    ).to(device)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    transforms = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT.transforms()
    return model, transforms




def ensure_look_structure(dataset: List[Dict]):
    """Ensure each look is a dict with Look_Number, Look_Url, Garments."""
    for designer in dataset:
        for show in designer.get("Shows", []):
            looks = show.get("Looks")
            if looks is None:
                continue
            for idx, look in enumerate(list(looks)):
                if isinstance(look, str):
                    looks[idx] = {
                        "Look_Number": idx + 1,
                        "Look_Url": look,
                        "Garments": {},
                    }
                elif isinstance(look, dict):
                    look.setdefault("Look_Number", idx + 1)
                    # try to find the url field if not present
                    look.setdefault("Look_Url", look.get("url") or look.get("image") or look.get("LookUrl") or "")
                    look.setdefault("Garments", {})


# ------------------------------------------------------------
# Concurrent image download
# ------------------------------------------------------------

def _fetch_image(task: Tuple[str, Dict, object]) -> Optional[Tuple[torch.Tensor, Dict]]:
    """Download one image and return (tensor, look_ref)."""
    url, look_ref, tfm = task
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
    except Exception as exc:
        print(f"⚠️  {url} → {exc}")
        return None

    return tfm(img), look_ref


# ------------------------------------------------------------
# Inference helpers
# ------------------------------------------------------------

def _infer_batch(model, img_list: List[torch.Tensor], meta_list: List[Dict], device: str):
    """Run one batch through the detector and update each look's Garments."""
    batch = [img.to(device) for img in img_list]
    with torch.no_grad():
        predictions = model(batch)

    for pred, look_ref in zip(predictions, meta_list):
        boxes = pred["boxes"].detach().cpu().numpy().tolist()
        labels = pred["labels"].detach().cpu().numpy().tolist()
        scores = pred["scores"].detach().cpu().numpy().tolist()
        look_ref["Garments"] = {
            "labels": labels,
            "boxes": boxes,
            "scores": scores,
        }


# ------------------------------------------------------------
# Main detection routine (threaded fetch + batched infer)
# ------------------------------------------------------------

def detect_objects(
    model,
    dataset: List[Dict],
    batch_size: int = 16,
    device: str = "cuda",
    tfm=None,
    max_workers: int = 16,
    start_letter: Optional[str] = None,
):
    """Download images concurrently, perform batched detection, and update dataset."""



    # Build tasks
    for designer in dataset:
        batch_imgs: List[torch.Tensor] = []
        batch_meta: List[Dict] = []
        tasks: List[Tuple[str, Dict, object]] = []
        if start_letter and not designer.get("Designer", "").startswith(start_letter):
            continue
        for show in designer.get("Shows", []):
            looks = show.get("Looks")
            if looks is None:
                continue
            for look in looks:
                url = look.get("Look_Url", "")
                if not url:
                    continue
                tasks.append((url, look, tfm))

    # Concurrent fetching
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [pool.submit(_fetch_image, t) for t in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Fetching"):
                result = future.result()
                if result is None:
                    continue
                tensor, look_ref = result
                batch_imgs.append(tensor)
                batch_meta.append(look_ref)
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
    model, transforms = get_model(num_classes=10, weights_path="weights/rcnn_finetune.pth", device=device)

    with open("fash-data/all_photos.json", "r") as f:
        dataset = json.load(f)

    dataset_out = copy.deepcopy(dataset)
    ensure_look_structure(dataset_out)

    detect_objects(
        model,
        dataset_out,
        batch_size=64,
        device=device,
        tfm=transforms,
        max_workers=16,
        start_letter="A",  # set to a letter like "A" to restrict
    )

    with open("fash-data/all_photos_with_garments.json", "w") as f:
        json.dump(dataset_out, f, indent=2)

    print("Saved updated dataset to fash-data/all_photos_with_garments.json")


if __name__ == "__main__":
    main()
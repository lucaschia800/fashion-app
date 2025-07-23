import json, torch
from PIL import Image
from tqdm import tqdm
import classes.multiheaded_Fefficient as mhf
from torchvision.models import EfficientNet_V2_M_Weights


# ----------------------------------------------------------------------
def get_model(ckpt_path, device="cuda"):
    model = mhf.MultiHead_FEfficientNet(ckpt_path=ckpt_path).to(device)
    model.eval()
    preprocess = EfficientNet_V2_M_Weights.DEFAULT.transforms()
    return model, preprocess


# ----------------------------------------------------------------------


def detect_attributes(
    model,
    dataset,
    letter,
    batch_size=16,
    device="cuda",
    preprocess=None,
):
    batch_imgs = []  # list[Tensor]  -- cropped tensors
    batch_meta = []  # list[dict]    -- where this crop came from

    for designer in tqdm(dataset, desc="Scanning dataset"):
        if not designer["Designer_Name"].startswith(letter):
            continue

        for show in designer["Shows"]:
            for look in show["Looks"]:
                img = Image.open(look["Look_Url"])
                boxes = look["Garments"]["boxes"]

                for box_idx, (x1, y1, x2, y2) in enumerate(boxes):
                    crop = img.crop((x1, y1, x2, y2))
                    crop = preprocess(crop)
                    batch_imgs.append(crop)
                    batch_meta.append(
                        {
                            "designer": designer["Designer_Name"],
                            "show_id": show["Show_Id"],
                            "look_id": look["Look_Id"],
                            "box_idx": box_idx,
                        }
                    )

                    if len(batch_imgs) == batch_size:
                        _infer_batch(model, batch_imgs, batch_meta, device)
                        batch_imgs.clear()
                        batch_meta.clear()

    # catch the tail
    if batch_imgs:
        _infer_batch(model, batch_imgs, batch_meta, device)


def _infer_batch(model, img_list, meta_list, device):
    """Helper: run one batch through the network and print/return results."""

    categories = {'gender' : 3, 'material' : 23, 'pattern' : 18, 'style' : 10, 'sleeve' : 4, 'category': 48, 'color' : 19}

    multiclass_categories = ['gender', 'style', 'sleeve']
    multilabel_categories = ['color', 'pattern', 'material', 'category']
    batch = torch.stack(img_list).to(device)  # [B,3,H,W]
    with torch.no_grad():
        logits = model(batch)  # dict[str, Tensor], each [B, C]

    # Example: just print – replace with whatever post-processing / saving you need
    for i, meta in enumerate(meta_list):
        print(
            f"\nCrop from {meta['designer']} – look {meta['look_id']} – box {meta['box_idx']}"
        )
        for attr, attr_logits in logits.items():
            pred = torch.sigmoid(attr_logits[i]).cpu().numpy()
            print(f"  {attr}: {pred}")


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, prep = get_model("weights/Mefficientnet_v2_freeze_ckpt2.pth", device=device)

    with open("letter_A.json") as f:
        dataset = json.load(f)

    detect_attributes(
        model, dataset, letter="A", batch_size=32, device=device, preprocess=prep
    )


if __name__ == "__main__":
    main()

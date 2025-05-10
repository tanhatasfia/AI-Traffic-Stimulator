import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import cv2
import tqdm
from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map

import crossView
from utils import mean_IU, mean_precision
from opt import get_eval_args as get_args


def load_model(models, model_path):
    model_path = os.path.expanduser(model_path)
    assert os.path.isdir(model_path), f"Cannot find folder {model_path}"
    print(f"Loading model from folder {model_path}")

    for key in models.keys():
        print(f"Loading {key} weights...")
        path = os.path.join(model_path, f"{key}.pth")
        model_dict = models[key].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[key].load_state_dict(model_dict)

    return models


def load_single_image(opt, img_path):
    from torchvision import transforms
    import PIL.Image as pil

    img = pil.open(img_path).convert("RGB")
    img = img.resize((opt.width, opt.height), pil.BILINEAR)
    img_tensor = transforms.ToTensor()(img).unsqueeze(0).to("cuda")

    return {"color": img_tensor, "filename": [os.path.splitext(os.path.basename(img_path))[0]]}


def process_batch(opt, models, inputs):
    outputs = {}
    for key, val in inputs.items():
        if key != "filename":
            inputs[key] = val.to("cuda")

    features = models["encoder"](inputs["color"])
    transform_feature, retransform_features = models["CycledViewProjection"](features)
    features = models["CrossViewTransformer"](features, transform_feature, retransform_features)

    outputs["topview"] = models["decoder"](features)
    outputs["transform_topview"] = models["transform_decoder"](transform_feature)

    return outputs


def save_topview(idx, tv, name_dest_im):
    tv_np = torch.argmax(tv.squeeze(), dim=0).cpu().numpy().astype(np.uint8)

    # 🚨 Add this line to debug
    print(f"{idx[0]} - Predicted unique values: {np.unique(tv_np)}")

    id2color = {
        0: (255, 255, 255),  # background
        1: (0, 0, 0),        # road
        2: (169, 169, 169),  # lane
        3: (253, 240, 1),    # vehicle
    }
    color_img = np.zeros((tv_np.shape[0], tv_np.shape[1], 3), dtype=np.uint8)
    for cls_id, color in id2color.items():
        color_img[tv_np == cls_id] = color

    os.makedirs(os.path.dirname(name_dest_im), exist_ok=True)
    cv2.imwrite(name_dest_im, color_img)



def evaluate():
    opt = get_args()

    # ==== Load Models ====
    models = {}
    models["encoder"] = crossView.Encoder(18, opt.height, opt.width, True)
    models["CycledViewProjection"] = crossView.CycledViewProjection(in_dim=8)
    models["CrossViewTransformer"] = crossView.CrossViewTransformer(128)
    models["decoder"] = crossView.Decoder(models["encoder"].resnet_encoder.num_ch_enc, opt.num_class)
    models["transform_decoder"] = crossView.Decoder(models["encoder"].resnet_encoder.num_ch_enc, opt.num_class, "transform_decoder")

    for key in models.keys():
        models[key].to("cuda")

    models = load_model(models, opt.pretrained_path)

    # ==== Load All Test Images from Folder ====
    image_dir = os.path.join(opt.data_path, "images")
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    print(f"✅ Found {len(image_files)} test images in: {image_dir}")

    for img_file in tqdm.tqdm(image_files):
        img_path = os.path.join(image_dir, img_file)
        inputs = load_single_image(opt, img_path)
        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)

        # === Save Predicted Image ===
        save_topview(
            [img_file],
            outputs["topview"],
            os.path.join(opt.out_dir, 'topview', f"{img_file}")
        )


if __name__ == "__main__":
    evaluate()

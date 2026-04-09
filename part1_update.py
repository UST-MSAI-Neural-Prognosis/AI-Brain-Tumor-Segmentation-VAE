
import sys
import subprocess
import os
import json
import random
import nibabel as nib
import numpy as np
import cv2

# --- SEG-01: Accessing Training Data ---
TRAIN_PATH = "/Users/Yue/Desktop/Project_766/BraTS2021_Training_Data"

def verify_data():
    if not os.path.exists(TRAIN_PATH):
        print(f"❌ Folder not found at: {TRAIN_PATH}")
        return []
    
    cases = [d for d in os.listdir(TRAIN_PATH) if d.startswith("BraTS2021")]
    print(f"📂 Found {len(cases)} training cases.")
    return cases

# --- SEG-02: Train/Val Split (Moved Up to Prevent Leakage) ---
def save_dataset_split(case_ids, train_ratio=0.8):
    """
    SEG-02: Split cases BEFORE any normalization or processing.
    """
    random.seed(42) 
    random.shuffle(case_ids)
    
    split_point = int(len(case_ids) * train_ratio)
    train_list = case_ids[:split_point]
    val_list = case_ids[split_point:]
    
    split_dict = {"train": train_list, "val": val_list}
    
    # --- UPDATED SECTION ---
    # Define the output path clearly. 
    # This puts the JSON on your Desktop folder where you have write access.
    output_path = "/Users/Yue/Desktop/Project_766/dataset_split.json"
    
    try:
        with open(output_path, "w") as f:
            json.dump(split_dict, f, indent=4)
        print(f"✅ Split saved to: {output_path}")
    except OSError as e:
        print(f"❌ Failed to save JSON. Error: {e}")
        # Fallback: Print to console so you don't lose the data
        print("Manual Split Data:", split_dict)
    # -----------------------
        
    print(f" - Training: {len(train_list)}, Validation: {len(val_list)}")
    return split_dict

# --- SEG-03: NIfTI loading pipeline ---
def load_case_data(case_id):
    case_path = os.path.join(TRAIN_PATH, case_id)
    modalities = ['t1', 't1ce', 't2', 'flair']
    data_dict = {}

    for mod in modalities:
        file_path = os.path.join(case_path, f"{case_id}_{mod}.nii.gz")
        img = nib.load(file_path)
        data_dict[mod] = img.get_fdata().astype(np.float32)

    seg_path = os.path.join(case_path, f"{case_id}_seg.nii.gz")
    seg_img = nib.load(seg_path)
    data_dict['seg'] = seg_img.get_fdata().astype(np.uint8)
    return data_dict

# --- SEG-04: Per-Volume Z-score Normalization ---
def normalize_volume(volume):
    """
    Applied on a per-case basis. Since MRI intensity scales vary between scanners,
    we normalize based on the individual's brain tissue.
    """
    brain_mask = volume > 0
    if np.any(brain_mask):
        mean = volume[brain_mask].mean()
        std = volume[brain_mask].std()
        normalized = (volume - mean) / (std + 1e-8)
        normalized[~brain_mask] = 0
        return normalized.astype(np.float32)
    return volume.astype(np.float32)

# --- SEG-05: Axial Slice Extraction ---
def extract_tumor_slices(data_dict, margin=5):
    seg = data_dict['seg']
    z_indices = np.any(seg > 0, axis=(0, 1))
    
    if not np.any(z_indices):
        return [], []

    start_z = max(0, np.where(z_indices)[0].min() - margin)
    end_z = min(seg.shape[2], np.where(z_indices)[0].max() + margin)

    slices_images, slices_masks = [], []
    for z in range(start_z, end_z):
        combined_slice = np.stack([
            data_dict['t1'][:, :, z],
            data_dict['t1ce'][:, :, z],
            data_dict['t2'][:, :, z],
            data_dict['flair'][:, :, z]
        ], axis=0) 
        slices_images.append(combined_slice)
        slices_masks.append(seg[:, :, z])
    return slices_images, slices_masks

# --- SEG-06: Paired Augmentation ---
def apply_paired_augmentation(image, mask):
    if random.random() > 0.5:
        image = np.flip(image, axis=2)
        mask = np.flip(mask, axis=1)

    angle = random.uniform(-15, 15)
    h, w = mask.shape
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

    aug_image = np.zeros_like(image)
    for c in range(4):
        aug_image[c] = cv2.warpAffine(image[c], rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)

    aug_mask = cv2.warpAffine(mask, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)
    aug_image = aug_image * random.uniform(0.9, 1.1)
    return aug_image, aug_mask

# --- Execution Logic ---
all_cases = verify_data()
if all_cases:
    # 1. SPLIT FIRST
    dataset_split = save_dataset_split(all_cases)
    
    # 2. PROCESS TRAINING DATA ONLY (Example with one case)
    target_case = dataset_split['train'][0]
    raw_data = load_case_data(target_case)
    
    # 3. NORMALIZE INDIVIDUALLY
    for mod in ['t1', 't1ce', 't2', 'flair']:
        raw_data[mod] = normalize_volume(raw_data[mod])
    
    # 4. EXTRACT AND AUGMENT
    imgs, masks = extract_tumor_slices(raw_data)
    if imgs:
        aug_img, aug_mask = apply_paired_augmentation(imgs[0], masks[0])
        print(f"🚀 Pipeline Complete for {target_case}. Ready for training.")
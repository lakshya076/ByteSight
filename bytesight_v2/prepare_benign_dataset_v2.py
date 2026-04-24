import os
import glob
import random
import numpy as np
import math
from PIL import Image
import multiprocessing
from functools import partial
import argparse
import shutil

def convert_to_png_lossless(input_path, output_path):
    """
    Converts a binary file to a 224x224 grayscale PNG image using zero-loss padding.
    """
    try:
        if not os.path.isfile(input_path):
            return False
            
        with open(input_path, 'rb') as f:
            data = f.read()

        if not data:
            return False

        arr = np.frombuffer(data, dtype=np.uint8)
        size = len(arr)
        
        # Target square size (next perfect square)
        side = int(math.ceil(math.sqrt(size)))
        if side == 0: side = 1
        
        # Calculate padding
        padding_needed = (side * side) - size
        if padding_needed > 0:
            padded_arr = np.pad(arr, (0, padding_needed), mode='constant', constant_values=0)
        else:
            padded_arr = arr
            
        img_array = padded_arr.reshape((side, side))
        img = Image.fromarray(img_array, 'L')
        
        # Resize using NEAREST to preserve binary texture
        img = img.resize((224, 224), Image.NEAREST)
        img.save(output_path)
        return True
    except Exception:
        return False

def worker_func(file_info, base_output_dir, total_found, split_ratio):
    idx, file_path = file_info
    
    # Calculate splits
    train_end = int(total_found * split_ratio[0])
    val_end = train_end + int(total_found * split_ratio[1])
    
    # Determine target split folder
    if idx < train_end:
        target_sub = 'train'
    elif idx < val_end:
        target_sub = 'val'
    else:
        target_sub = 'test'
        
    target_dir = os.path.join(base_output_dir, target_sub, 'Benign')
    os.makedirs(target_dir, exist_ok=True)
    
    # Use index and filename to ensure uniqueness
    output_filename = f"benign_{idx}.png"
    output_path = os.path.join(target_dir, output_filename)
    
    return convert_to_png_lossless(file_path, output_path)

def prepare_benign_dataset(input_dirs, base_output_dir, limit=10868, cores=None):
    """
    Collects binaries from multiple directories and converts them.
    """
    # SAFETY: Purge old benign data to prevent leakage/duplicates
    for split in ['train', 'val', 'test']:
        old_path = os.path.join(base_output_dir, split, 'Benign')
        if os.path.exists(old_path):
            print(f"[*] Purging old benign data in {old_path}...")
            shutil.rmtree(old_path)

    split_ratio = (0.7, 0.15, 0.15)
    all_candidates = []
    
    print(f"[*] Scanning for benign Windows binaries...")
    
    extensions = ['.exe', '.dll', '.sys', '.bin', '.ocx', '.cpl']

    for root_dir in input_dirs:
        if not os.path.exists(root_dir):
            continue
        print(f"[*] Scanning: {root_dir}")
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if any(file.lower().endswith(ext) for ext in extensions):
                    file_path = os.path.join(root, file)
                    if os.path.isfile(file_path):
                        all_candidates.append(file_path)
            
            if len(all_candidates) >= limit * 1.5:
                break
        if len(all_candidates) >= limit * 1.5:
            break

    if not all_candidates:
        print("Error: No Windows binaries found. Check if /mnt/c is mounted correctly.")
        return

    random.shuffle(all_candidates)
    selected_files = all_candidates[:limit]
    total_found = len(selected_files)
    
    print(f"[*] Found {len(all_candidates)} candidates, processing {total_found} samples.")
    
    if cores is None:
        cores = multiprocessing.cpu_count()
        
    print(f"[*] Starting conversion using {cores} cores...")
    
    work_items = list(enumerate(selected_files))
    process_func = partial(worker_func, 
                          base_output_dir=base_output_dir, 
                          total_found=total_found, 
                          split_ratio=split_ratio)
    
    processed_count = 0
    with multiprocessing.Pool(cores) as pool:
        for result in pool.imap_unordered(process_func, work_items):
            if result:
                processed_count += 1
                if processed_count % 500 == 0:
                    print(f"Processed {processed_count}/{total_found} files...")

    print(f"\n--- Benign Preparation Complete ---")
    print(f"Total files converted: {processed_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a Benign class dataset using v2 lossless logic.")
    parser.add_argument("--inputs", nargs="+", default=["/mnt/c/Windows/System32", "/mnt/c/Windows/SysWOW64"], 
                        help="List of directories to scan")
    parser.add_argument("--output", default="../microsoft_dataset", help="Dataset directory")
    parser.add_argument("--limit", type=int, default=10868, help="Number of files to collect")
    parser.add_argument("--cores", type=int, help="Number of CPU cores to use")
    
    args = parser.parse_args()
    prepare_benign_dataset(args.inputs, args.output, args.limit, args.cores)

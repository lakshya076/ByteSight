import os
import csv
import subprocess
import numpy as np
import math
from PIL import Image
import io
import multiprocessing
from functools import partial
import argparse

# Malware family mapping for Microsoft Challenge
CLASS_NAMES = {
    "1": "Ramnit",
    "2": "Lollipop",
    "3": "Kelihos_ver3",
    "4": "Vundo",
    "5": "Simda",
    "6": "Tracur",
    "7": "Kelihos_ver1",
    "8": "Obfuscator.ACY",
    "9": "Gatak"
}

def process_bytes_stream(stream):
    """
    Parses the .bytes hex dump stream and returns a numpy array of bytes.
    The .bytes files are text files with hex values.
    Example: 00401000 56 8D 44 24 08 50 8D 44 24 1C 50 E8 F3 09 00 00
    """
    byte_list = []
    # Use a buffered reader for efficiency
    reader = io.BufferedReader(stream)
    for line in reader:
        try:
            line_str = line.decode('utf-8')
            parts = line_str.split()
            if not parts:
                continue
            # parts[0] is the address, parts[1:] are hex bytes
            for hex_val in parts[1:]:
                if hex_val == '??':
                    byte_list.append(0)
                else:
                    byte_list.append(int(hex_val, 16))
        except Exception:
            continue
            
    return np.array(byte_list, dtype=np.uint8)

def save_as_png(arr, output_path):
    """
    Converts byte array to square image and resizes to 224x224.
    Uses padding instead of trimming to preserve 100% of data.
    """
    if len(arr) == 0:
        return
    
    size = len(arr)
    # Target square size (next perfect square)
    side = int(math.ceil(math.sqrt(size)))
    if side == 0: side = 1
    
    # Calculate padding needed
    padding_needed = (side * side) - size
    
    if padding_needed > 0:
        # Pad with zeros
        padded_arr = np.pad(arr, (0, padding_needed), mode='constant', constant_values=0)
    else:
        padded_arr = arr
        
    img_array = padded_arr.reshape((side, side))
    
    img = Image.fromarray(img_array, 'L')
    # Resize to fixed dimension for CNN (ResNet)
    img = img.resize((224, 224), Image.NEAREST)
    img.save(output_path)

def process_single_file(file_info, archive_path, local_train_dir, output_base, split_ratio, total_count):
    """
    Worker function to process one file.
    """
    idx, file_path, class_id = file_info
    file_id = os.path.basename(file_path).replace(".bytes", "")
    
    # Determine split
    train_end = int(total_count * split_ratio[0])
    val_end = train_end + int(total_count * split_ratio[1])
    
    if idx < train_end:
        split = 'train'
    elif idx < val_end:
        split = 'val'
    else:
        split = 'test'
        
    class_name = CLASS_NAMES.get(class_id, f"Class_{class_id}")
    target_dir = os.path.join(output_base, split, class_name)
    os.makedirs(target_dir, exist_ok=True)
    
    output_path = os.path.join(target_dir, f"{file_id}.png")
    
    if os.path.exists(output_path):
        return True

    # Check if file exists locally first
    local_file_path = os.path.join(local_train_dir, os.path.basename(file_path))
    
    try:
        if os.path.exists(local_file_path):
            # Fast Path: Read from disk
            with open(local_file_path, 'rb') as f:
                arr = process_bytes_stream(f)
        else:
            # Slow Path: Stream from archive
            cmd = ["7z", "x", archive_path, "-so", file_path]
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            arr = process_bytes_stream(proc.stdout)
            proc.wait()
        
        if len(arr) > 0:
            save_as_png(arr, output_path)
            return True
    except Exception as e:
        print(f"Error processing {file_id}: {e}")
        
    return False

def get_ordered_files(archive_path):
    """
    Returns the list of .bytes files in the order they appear in the archive.
    This is crucial for performance with solid 7z archives.
    """
    print("Fetching file list from archive (this might take a moment)...")
    cmd = ["7z", "l", archive_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    files = []
    for line in result.stdout.splitlines():
        if line.endswith(".bytes"):
            # The filename is the last part of the line
            parts = line.split()
            if parts:
                files.append(parts[-1])
    return files

def main():
    parser = argparse.ArgumentParser(description="Convert Microsoft Malware Challenge dataset to PNGs.")
    parser.add_argument("--archive", default="../train.7z", help="Path to train.7z")
    parser.add_argument("--local_dir", default="../train", help="Path to extracted .bytes files")
    parser.add_argument("--labels", default="../trainLabels.csv", help="Path to trainLabels.csv")
    parser.add_argument("--output", default="../microsoft_dataset", help="Output directory")
    parser.add_argument("--limit", type=int, help="Limit total files processed")
    parser.add_argument("--cores", type=int, default=multiprocessing.cpu_count(), help="Number of cores to use")
    
    args = parser.parse_args()

    archive_path = args.archive
    local_train_dir = args.local_dir
    labels_csv = args.labels
    output_base = args.output
    limit = args.limit
    split_ratio = (0.7, 0.15, 0.15) # train, val, test
    num_cores = args.cores
    
    if not os.path.exists(archive_path) and not os.path.exists(local_train_dir):
        print(f"Error: Neither {archive_path} nor {local_train_dir} folder found.")
        return
    
    # 1. Load labels
    print("Loading labels...")
    id_to_class = {}
    with open(labels_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            id_to_class[row['Id']] = row['Class']
    
    # 2. Get ordered files (prefer archive order for streaming, or local list)
    if os.path.exists(archive_path):
        ordered_files = get_ordered_files(archive_path)
    else:
        print("Archive not found, using local directory file list...")
        ordered_files = [os.path.join(local_train_dir, f) for f in os.listdir(local_train_dir) if f.endswith(".bytes")]
    
    # 3. Filter and Limit
    valid_files = [f for f in ordered_files if os.path.basename(f).replace(".bytes", "") in id_to_class]
    if limit:
        valid_files = valid_files[:limit]
    
    total_to_process = len(valid_files)
    
    # 4. Prepare work items
    work_items = []
    for i, file_path in enumerate(valid_files):
        file_id = os.path.basename(file_path).replace(".bytes", "")
        class_id = id_to_class.get(file_id)
        work_items.append((i, file_path, class_id))
    
    # 5. Multiprocessing Pool
    print(f"Starting processing of {total_to_process} files using {num_cores} cores...")
    
    # Use partial to fix constant arguments
    worker_func = partial(process_single_file, 
                         archive_path=archive_path, 
                         local_train_dir=local_train_dir, 
                         output_base=output_base, 
                         split_ratio=split_ratio, 
                         total_count=total_to_process)
    
    processed_count = 0
    with multiprocessing.Pool(num_cores) as pool:
        for result in pool.imap_unordered(worker_func, work_items):
            if result:
                processed_count += 1
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count}/{total_to_process} files...")

    print(f"\nFinished! Processed {processed_count} files.")
    print(f"Images are stored in: {output_base}")

if __name__ == "__main__":
    main()

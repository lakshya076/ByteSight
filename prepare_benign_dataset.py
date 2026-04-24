import os
import glob
import random
import shutil
from converter import convert_to_png

def prepare_benign_dataset(input_dir, base_output_dir, limit=600, split_ratio=(0.7, 0.15, 0.15)):
    """
    Collects binaries from input_dir, converts them to images, and splits them into
    train, val, and test folders.
    """
    # 1. Setup Folder Structure
    subfolders = ['train', 'val', 'test']
    for folder in subfolders:
        os.makedirs(os.path.join(base_output_dir, folder, 'Benign'), exist_ok=True)

    # 2. Find Candidate Files
    # Common Windows binary extensions
    extensions = ['*.exe', '*.dll', '*.sys', '*.bin']
    all_files = []
    for ext in extensions:
        all_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    # If no files found with extensions, try everything (useful for Linux /usr/bin)
    if not all_files:
        all_files = [f for f in glob.glob(os.path.join(input_dir, "*")) if os.path.isfile(f)]

    # Shuffle to get a random mix
    random.shuffle(all_files)
    
    # Respect the limit
    selected_files = all_files[:limit]
    total_found = len(selected_files)
    
    if total_found == 0:
        print(f"Error: No binary files found in {input_dir}")
        return

    # 3. Calculate Splits
    train_end = int(total_found * split_ratio[0])
    val_end = train_end + int(total_found * split_ratio[1])

    # 4. Process and Convert
    print(f"Starting conversion of {total_found} files...")
    
    for i, file_path in enumerate(selected_files):
        # Determine target folder based on index
        if i < train_end:
            target_sub = 'train'
        elif i < val_end:
            target_sub = 'val'
        else:
            target_sub = 'test'
            
        filename = os.path.basename(file_path)
        output_filename = f"benign_{i}.png"
        output_path = os.path.join(base_output_dir, target_sub, 'Benign', output_filename)
        
        try:
            convert_to_png(file_path, output_path)
            if (i + 1) % 50 == 0:
                print(f"Processed {i + 1}/{total_found} files...")
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")

    print("\n--- Preparation Complete ---")
    print(f"Total files converted: {i+1}")
    print(f"Location: {base_output_dir}")
    print(f"Structure: Benign/{{train, val, test}}/Benign/")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build a Benign class dataset from local binaries.")
    
    # Defaults for a Windows environment (can be overridden for Linux)
    parser.add_argument("--input", default="C:\\Windows\\System32", help="Path to binaries (e.g., C:\\Windows\\System32)")
    parser.add_argument("--output", default="malimg_dataset", help="Base dataset directory")
    parser.add_argument("--limit", type=int, default=500, help="Total number of files to collect")
    
    args = parser.parse_args()
    
    # Convert input path to handle OS-specific slashes
    input_path = os.path.normpath(args.input)
    
    prepare_benign_dataset(input_path, args.output, args.limit)

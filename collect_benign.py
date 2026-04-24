import os
import glob
from converter import convert_to_png

def collect_benign_samples(input_dir, output_dir, limit=500):
    """
    Finds binary files in input_dir and converts them to PNGs in output_dir.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Find files (on Linux we look for all files in bin, on Windows we'd look for .exe)
    # This glob handles both common scenarios
    files = glob.glob(os.path.join(input_dir, "*")) + glob.glob(os.path.join(input_dir, "*.exe"))
    
    count = 0
    for file_path in files:
        if count >= limit:
            break
            
        # Skip directories
        if os.path.isdir(file_path):
            continue
            
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, f"benign_{count}.png")
        
        try:
            # We use the conversion logic from our existing converter.py
            convert_to_png(file_path, output_path)
            count += 1
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    print(f"\nFinished! Collected {count} benign samples in {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Collect and convert benign binaries for training.")
    # Defaulting to /usr/bin for Linux environments. 
    # For Windows, you'd use "C:\\Windows\\System32"
    parser.add_argument("--input", default="/usr/bin", help="Source directory for benign binaries")
    parser.add_argument("--output", default="malimg_dataset/train/Benign", help="Target directory for images")
    parser.add_argument("--limit", type=int, default=100, help="Max number of files to convert")
    
    args = parser.parse_args()
    collect_benign_samples(args.input, args.output, args.limit)

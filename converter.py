import os
import math
import numpy as np
from PIL import Image
import argparse

def convert_to_png(input_path, output_path):
    """
    Converts a binary file to a 224x224 grayscale PNG image.
    """
    if not os.path.exists(input_path):
        print(f"Error: File {input_path} not found.")
        return

    # 1. Read binary data
    with open(input_path, 'rb') as f:
        data = f.read()

    if not data:
        print(f"Error: File {input_path} is empty.")
        return

    # 2. Convert to numpy array
    arr = np.frombuffer(data, dtype=np.uint8)
    
    # 3. Determine dimensions
    # For a demo, we'll create a square as close as possible, 
    # then resize to 224x224 to match the model's input.
    size = len(arr)
    width = int(math.sqrt(size))
    if width == 0: width = 1
    
    # Trim array to fit into a perfect square
    trimmed_arr = arr[:width*width]
    
    # Reshape to 2D
    img_array = trimmed_arr.reshape((width, width))

    # 4. Create Image object
    img = Image.fromarray(img_array, 'L') # 'L' means grayscale

    # 5. Resize to match ResNet18 input (224x224)
    # Using NEAREST resampling to keep the "texture" sharp
    img = img.resize((224, 224), Image.NEAREST)

    # 6. Save
    img.save(output_path)
    print(f"Success! Converted {input_path} ({len(data)} bytes) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert any binary to a ByteSight-ready image.")
    parser.add_argument("input", help="Path to the .exe or binary file")
    parser.add_argument("--output", default="demo_output.png", help="Output PNG filename")
    
    args = parser.parse_args()
    convert_to_png(args.input, args.output)

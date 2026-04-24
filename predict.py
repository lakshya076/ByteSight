import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F

# 1. Define the class names (extracted from the dataset)
class_names = [
    'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J',
    'Autorun.K', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 'Dontovo.A',
    'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 'Lolyda.AA3',
    'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 'Skintrim.N',
    'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 'Yuner.A'
]

def predict(image_path):
    # 2. Re-create the architecture
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 25)
    
    # Check if trained weights exist and load them
    weights_path = 'bytesight_resnet_prototype.pth'
    import os
    if os.path.exists(weights_path):
        # Load weights, handling potential CPU/GPU mapping
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        weights_loaded = True
    else:
        weights_loaded = False
    
    model.eval()

    # 3. Define the same transforms as used in training
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 4. Load and transform the image
    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension

    # 5. Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        
        # Get the top class and its confidence
        conf, index = torch.max(probabilities, 1)
        
    class_name = class_names[index.item()]
    confidence = conf.item() * 100

    print(f"\n--- ByteSight Prediction ---")
    print(f"Image: {image_path}")
    print(f"Predicted Class: {class_name}")
    print(f"Confidence Score: {confidence:.2f}%")
    print(f"----------------------------\n")
    
    if weights_loaded:
        print(f"Status: Using trained weights from '{weights_path}'")
    else:
        print("Status: Using random initialization (Trained weights not found).")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ByteSight prediction on an image.")
    parser.add_argument("image", help="Path to the image file (PNG/JPG)")
    
    args = parser.parse_args()
    predict(args.image)

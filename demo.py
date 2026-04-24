import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import torch.nn.functional as F
import os
import random
import glob

# Define the class names
class_names = [
    'Adialer.C', 'Agent.FYI', 'Allaple.A', 'Allaple.L', 'Alueron.gen!J',
    'Autorun.K', 'Benign', 'C2LOP.P', 'C2LOP.gen!g', 'Dialplatform.B', 
    'Dontovo.A', 'Fakerean', 'Instantaccess', 'Lolyda.AA1', 'Lolyda.AA2', 
    'Lolyda.AA3', 'Lolyda.AT', 'Malex.gen!J', 'Obfuscator.AD', 'Rbot!gen', 
    'Skintrim.N', 'Swizzor.gen!E', 'Swizzor.gen!I', 'VB.AT', 'Wintrim.BX', 
    'Yuner.A'
]

def load_model():
    model = models.resnet18()
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 26)
    
    weights_path = 'bytesight_resnet_prototype.pth'
    if os.path.exists(weights_path):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.load_state_dict(torch.load(weights_path, map_location=device))
        print(f"[*] Loaded trained weights from {weights_path}")
    else:
        print("[!] Warning: Using randomly initialized model (weights not found).")
    
    model.eval()
    return model

def run_demo(temperature=1.0):
    model = load_model()
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    
    test_dir = 'malimg_dataset/test'
    
    # Collect Benign and Malware samples
    benign_samples = glob.glob(os.path.join(test_dir, 'Benign', '*.png'))
    all_malware_dirs = [d for d in glob.glob(os.path.join(test_dir, '*')) if os.path.isdir(d) and 'Benign' not in d]
    malware_samples = []
    for d in all_malware_dirs:
        malware_samples.extend(glob.glob(os.path.join(d, '*.png')))
    
    # Pick 3 benign and 7 malware
    if len(benign_samples) < 3:
        selected_benign = benign_samples
    else:
        selected_benign = random.sample(benign_samples, 3)
        
    num_malware_needed = 10 - len(selected_benign)
    selected_malware = random.sample(malware_samples, num_malware_needed)
    
    test_set = selected_benign + selected_malware
    random.shuffle(test_set)
    
    print(f"\n{'#'*60}")
    print(f"{'ByteSight Hackathon Demo':^60}")
    print(f"{f'Temperature: {temperature} (Calibration Active)':^60}")
    print(f"{'#'*60}\n")
    print(f"{'Original Class':<20} | {'Predicted Class':<20} | {'Confidence':<10}")
    print("-" * 60)
    
    for img_path in test_set:
        original_class = os.path.basename(os.path.dirname(img_path))
        
        image = Image.open(img_path)
        image_tensor = transform(image).unsqueeze(0)
        
        with torch.no_grad():
            logits = model(image_tensor)
            
            scaled_logits = logits / temperature
            probabilities = F.softmax(scaled_logits, dim=1)
            
            conf, index = torch.max(probabilities, 1)
        
        predicted_class = class_names[index.item()]
        confidence = conf.item() * 100
        
        result_marker = "[OK]" if original_class == predicted_class else "[!!]"
        print(f"{original_class:<20} | {predicted_class:<20} | {confidence:>8.2f}%  {result_marker}")

    print("\n" + "#"*60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ByteSight demo with Temperature Scaling.")
    parser.add_argument("-t", "--temperature", type=float, default=1.0, 
                        help="Temperature for Softmax (T > 1.0 softens/calibrates confidence)")
    
    args = parser.parse_args()
    run_demo(temperature=args.temperature)

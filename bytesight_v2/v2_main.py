import os
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import numpy as np
import math
import random
import glob
from PIL import Image
from torchvision import transforms
from models.resnet_model import get_model
from dataset import get_dataloaders
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Import Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    from pytorch_grad_cam.utils.image import show_cam_on_image
    HAS_GRADCAM = True
except ImportError:
    HAS_GRADCAM = False

def binary_to_tensor(file_path, img_size=224):
    """
    Converts a raw binary file to a PyTorch tensor using lossless padding.
    """
    with open(file_path, 'rb') as f:
        data = f.read()
    
    if not data:
        raise ValueError("File is empty.")

    arr = np.frombuffer(data, dtype=np.uint8)
    size = len(arr)
    side = int(math.ceil(math.sqrt(size)))
    if side == 0: side = 1
    
    padding_needed = (side * side) - size
    if padding_needed > 0:
        padded_arr = np.pad(arr, (0, padding_needed), mode='constant', constant_values=0)
    else:
        padded_arr = arr
        
    img_array = padded_arr.reshape((side, side))
    img = Image.fromarray(img_array, 'L')
    img = img.resize((img_size, img_size), Image.NEAREST)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0), img

def infer_pipeline(detector, classifier, family_classes, input_path, device, backbone, temperature=1.0):
    """
    Runs a file through the 2-stage hierarchical pipeline with temperature scaling.
    Returns: (is_malware, predicted_class, confidence, original_pil)
    """
    if input_path.lower().endswith(('.png', '.jpg', '.jpeg')):
        img = Image.open(input_path).convert('L')
        img = img.resize((224, 224), Image.NEAREST)
        input_tensor = transforms.ToTensor()(img).unsqueeze(0)
        original_pil = img
    else:
        input_tensor, original_pil = binary_to_tensor(input_path)
        
    input_tensor = input_tensor.to(device)
    
    # Stage 1
    with torch.no_grad():
        outputs = detector(input_tensor)
        probs = torch.nn.functional.softmax(outputs / temperature, dim=1)
        conf, index = torch.max(probs, 1)
        
    is_malware = (index.item() == 1)
    
    if not is_malware:
        return False, "Benign", conf.item() * 100, original_pil, input_tensor

    # Stage 2
    if classifier is None:
        return True, "Malware (Unknown)", conf.item() * 100, original_pil, input_tensor

    with torch.no_grad():
        outputs = classifier(input_tensor)
        probs = torch.nn.functional.softmax(outputs / temperature, dim=1)
        conf, index = torch.max(probs, 1)
        
    family = family_classes[index.item()]
    return True, family, conf.item() * 100, original_pil, input_tensor

def run_demo(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models once
    detector_weights = os.path.join(args.checkpoint_dir, 'binary_model.pth')
    if not os.path.exists(detector_weights):
        print(f"Error: Stage 1 model not found.")
        return

    detector = get_model(num_classes=2, backbone=args.backbone, pretrained=False)
    detector.load_state_dict(torch.load(detector_weights, map_location=device))
    detector.to(device).eval()

    classifier_weights = os.path.join(args.checkpoint_dir, 'malware_only_model.pth')
    class_file = os.path.join(args.checkpoint_dir, 'malware_only_classes.txt')
    
    classifier = None
    family_classes = []
    if os.path.exists(classifier_weights) and os.path.exists(class_file):
        with open(class_file, 'r') as f:
            family_classes = [line.strip() for line in f.readlines()]
        classifier = get_model(num_classes=len(family_classes), backbone=args.backbone, pretrained=False)
        classifier.load_state_dict(torch.load(classifier_weights, map_location=device))
        classifier.to(device).eval()

    # 2. Collect test samples
    test_dir = os.path.join(args.data_dir, 'test')
    if not os.path.exists(test_dir):
        print(f"Error: Test directory {test_dir} not found.")
        return

    benign_samples = glob.glob(os.path.join(test_dir, 'Benign', '*.png'))
    malware_samples = []
    for d in glob.glob(os.path.join(test_dir, '*')):
        if os.path.isdir(d) and 'Benign' not in d:
            malware_samples.extend(glob.glob(os.path.join(d, '*.png')))

    num_benign = max(1, int(args.num_samples * 0.3))
    num_malware = args.num_samples - num_benign

    selected_benign = random.sample(benign_samples, min(len(benign_samples), num_benign))
    selected_malware = random.sample(malware_samples, min(len(malware_samples), num_malware))
    
    test_set = selected_benign + selected_malware
    random.shuffle(test_set)

    # 3. Process and Print Table
    print(f"\n{'#'*85}")
    print(f"{'ByteSight v2 Hierarchical Demo':^85}")
    if args.temperature != 1.0:
        print(f"{f'Temperature: {args.temperature} (Calibration Active)':^85}")
    print(f"{'#'*85}\n")
    print(f"{'Original Class':<20} | {'Predicted Class':<20} | {'Conf':<10} | {'Result'}")
    print("-" * 85)

    correct = 0
    for path in test_set:
        original_class = os.path.basename(os.path.dirname(path))
        is_mw, predicted, conf, _, _ = infer_pipeline(detector, classifier, family_classes, path, device, args.backbone, temperature=args.temperature)
        
        # Result logic: In 2-stage, it's correct if the family matches OR if both are benign
        is_correct = (original_class == predicted)
        if is_correct: correct += 1
        
        marker = "[OK]" if is_correct else "[!!]"
        print(f"{original_class:<20} | {predicted:<20} | {conf:>8.2f}% | {marker}")

    print("-" * 85)
    print(f"OVERALL ACCURACY: {(correct/len(test_set))*100:.2f}% ({correct}/{len(test_set)})")
    print(f"{'#'*85}\n")

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on {device} in MODE: {args.mode}")
    
    dataloaders, class_names = get_dataloaders(args.data_dir, batch_size=args.batch_size, mode=args.mode)
    num_classes = len(class_names)
    print(f"[*] Found {num_classes} classes: {class_names}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    class_file = f"{args.mode}_classes.txt"
    with open(os.path.join(args.checkpoint_dir, class_file), 'w') as f:
        f.write('\n'.join(class_names))
    
    model = get_model(num_classes=num_classes, backbone=args.backbone, pretrained=True)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    best_acc = 0.0
    checkpoint_name = f"{args.mode}_model.pth"
    
    if not args.test_only:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            for phase in ['train', 'val']:
                if phase not in dataloaders: continue
                model.train() if phase == 'train' else model.eval()
                
                running_loss, running_corrects = 0.0, 0
                for inputs, labels in dataloaders[phase]:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
                
                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, checkpoint_name))
                    print(f"[*] Saved best {args.mode} model checkpoint")

    if 'test' in dataloaders:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_dir, checkpoint_name), map_location=device))
        # Re-import to avoid conflict
        from v2_main import evaluate_model
        evaluate_model(model, dataloaders['test'], class_names, device, args.checkpoint_dir, args.mode)

def evaluate_model(model, dataloader, class_names, device, checkpoint_dir, mode):
    model.eval()
    all_preds, all_labels = [], []
    print(f"[*] Running final evaluation for {mode}...")
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(f"\nFinal Test Accuracy: {acc*100:.2f}%\n\nClassification Report:\n{report}")
    
    with open(os.path.join(checkpoint_dir, f'{mode}_evaluation_report.txt'), 'w') as f:
        f.write(f"Final Test Accuracy: {acc*100:.2f}%\n\nClassification Report:\n{report}")

    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, xticks_rotation='vertical')
    plt.savefig(os.path.join(checkpoint_dir, f'{mode}_confusion_matrix.png'))
    plt.close()

def predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Models
    detector_weights = os.path.join(args.checkpoint_dir, 'binary_model.pth')
    detector = get_model(num_classes=2, backbone=args.backbone, pretrained=False)
    detector.load_state_dict(torch.load(detector_weights, map_location=device))
    detector.to(device).eval()

    classifier_weights = os.path.join(args.checkpoint_dir, 'malware_only_model.pth')
    class_file = os.path.join(args.checkpoint_dir, 'malware_only_classes.txt')
    classifier = None
    family_classes = []
    if os.path.exists(classifier_weights) and os.path.exists(class_file):
        with open(class_file, 'r') as f:
            family_classes = [line.strip() for line in f.readlines()]
        classifier = get_model(num_classes=len(family_classes), backbone=args.backbone, pretrained=False)
        classifier.load_state_dict(torch.load(classifier_weights, map_location=device))
        classifier.to(device).eval()

    # 2. Run Pipeline
    is_mw, predicted, conf, original_pil, input_tensor = infer_pipeline(detector, classifier, family_classes, args.input, device, args.backbone, temperature=args.temperature)
    
    print(f"\n{'='*40}")
    print(f"RESULT: {predicted}")
    print(f"CONFIDENCE: {conf:.2f}%")
    print(f"{'='*40}")
    
    if args.gradcam:
        target_model = classifier if is_mw and classifier else detector
        suffix = "malware" if is_mw else "safe"
        from v2_main import generate_gradcam
        generate_gradcam(target_model, input_tensor, original_pil, f"explanation_{suffix}.png", target_model.model.layer4[-1])

def generate_gradcam(model, input_tensor, original_img, output_path, target_layer):
    if not HAS_GRADCAM: return
    cam = GradCAM(model=model, target_layers=[target_layer])
    grayscale_cam = cam(input_tensor=input_tensor)[0, :]
    img_rgb = np.array(original_img.convert('RGB')).astype(np.float32) / 255.0
    visualization = show_cam_on_image(img_rgb, grayscale_cam, use_rgb=True)
    plt.imsave(output_path, visualization)
    print(f"[*] Grad-CAM saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="ByteSight v2 CLI")
    subparsers = parser.add_subparsers(dest="command")
    
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--data_dir", default="../microsoft_dataset")
    train_parser.add_argument("--mode", choices=['binary', 'malware_only', 'family'], default='binary')
    train_parser.add_argument("--epochs", type=int, default=10)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--lr", type=float, default=0.001)
    train_parser.add_argument("--backbone", default="resnet18")
    train_parser.add_argument("--checkpoint_dir", default="checkpoints")
    train_parser.add_argument("--test_only", action="store_true")
    
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--checkpoint_dir", default="checkpoints")
    predict_parser.add_argument("--backbone", default="resnet18")
    predict_parser.add_argument("--gradcam", action="store_true")
    predict_parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Temperature for Softmax calibration")

    demo_parser = subparsers.add_parser("demo")
    demo_parser.add_argument("--num_samples", type=int, default=10)
    demo_parser.add_argument("--data_dir", default="../microsoft_dataset")
    demo_parser.add_argument("--checkpoint_dir", default="checkpoints")
    demo_parser.add_argument("--backbone", default="resnet18")
    demo_parser.add_argument("-t", "--temperature", type=float, default=1.0, help="Temperature for Softmax calibration")
    
    args = parser.parse_args()
    if args.command == "train":
        train(args)
    elif args.command == "predict":
        predict(args)
    elif args.command == "demo":
        run_demo(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

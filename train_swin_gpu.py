import os
import sys
import cv2
import json
import time
import numpy as np
import pandas as pd

# Force unbuffered console output for real-time terminal updates
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import matplotlib
matplotlib.use('Agg')  # Headless backend prevents Windows GUI blocks
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import timm  # PyTorch Image Models library for Swin Transformer

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# ==========================================
# 1. HARDWARE VERIFICATION
# ==========================================
print("=" * 65)
print("INITIALIZING SWIN TRANSFORMER GPU TRAINING PIPELINE")
print("=" * 65)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Target Compute Device : {device}")

if torch.cuda.is_available():
    print(f"Detected GPU Model    : {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory Available : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("WARNING: CUDA is not active! Running on CPU.")

# ==========================================
# 2. DATASET & QUALITY ENHANCER
# ==========================================
CLASS_PATHS = {
    "Anthracnose": r"C:\Users\E028.10\Documents\Manish_trainigs\sorghum_dataset\Anthracnose",
    "Aphid": r"C:\Users\E028.10\Documents\Manish_trainigs\sorghum_dataset\Aphid",
    "Downey_mildew": r"C:\Users\E028.10\Documents\Manish_trainigs\sorghum_dataset\Downey_mildew",
    "Fall_army_worm": r"C:\Users\E028.10\Documents\Manish_trainigs\sorghum_dataset\Fall_army_worm",
    "Healthy": r"C:\Users\E028.10\Documents\Manish_trainigs\sorghum_dataset\Healthy"
}

class ImageQualityEnhancer:
    def __init__(self, blur_threshold=100.0, contrast_threshold=40.0):
        self.blur_threshold = blur_threshold
        self.contrast_threshold = contrast_threshold
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def compute_scores(self, gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var(), gray.std()

    def enhance(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        blur_score, contrast_score = self.compute_scores(gray)
        enhanced = img_bgr.copy()

        if contrast_score < self.contrast_threshold:
            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l = self.clahe.apply(l)
            enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

        if blur_score < self.blur_threshold:
            gb = cv2.GaussianBlur(enhanced, (0, 0), 3)
            enhanced = cv2.addWeighted(enhanced, 1.5, gb, -0.5, 0)

        return enhanced

enhancer = ImageQualityEnhancer()

class CachedSorghumDataset(Dataset):
    def __init__(self, samples, is_train=True):
        self.is_train = is_train
        self.cached_tensors = []
        
        mode = "Training" if is_train else "Validation"
        total = len(samples)
        print(f"\nPre-processing & caching {total} {mode} images into RAM...", flush=True)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        for i, (path, label) in enumerate(samples):
            if (i + 1) % 500 == 0 or (i + 1) == total:
                percent = ((i + 1) / total) * 100
                print(f" -> Caching {mode}: {i + 1}/{total} images ({percent:.1f}%)", flush=True)

            img_bgr = cv2.imread(path)
            if img_bgr is None:
                continue
            enhanced_bgr = enhancer.enhance(img_bgr)
            img_rgb = cv2.cvtColor(enhanced_bgr, cv2.COLOR_BGR2RGB)
            img_rgb = cv2.resize(img_rgb, (224, 224))

            tensor_img = torch.tensor(img_rgb, dtype=torch.float32).permute(2, 0, 1) / 255.0
            normalized = (tensor_img - mean) / std
            self.cached_tensors.append((normalized, label))

    def __len__(self):
        return len(self.cached_tensors)

    def __getitem__(self, idx):
        img, label = self.cached_tensors[idx]
        if self.is_train and np.random.rand() > 0.5:
            img = torch.flip(img, [2])
        return img, label

# Load Data Paths
print("\nScanning dataset directories...", flush=True)
all_samples = []
classes = list(CLASS_PATHS.keys())
class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

for cls_name, path in CLASS_PATHS.items():
    if not os.path.exists(path):
        print(f"Warning: Folder missing -> {path}", flush=True)
        continue
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                all_samples.append((os.path.join(root, f), class_to_idx[cls_name]))

print(f"Total Dataset Images Found: {len(all_samples)}", flush=True)

# 80/20 Train/Validation Split
np.random.shuffle(all_samples)
split = int(len(all_samples) * 0.8)
train_samples, val_samples = all_samples[:split], all_samples[split:]

print(f"Training set split   : {len(train_samples)} samples (80%)")
print(f"Validation set split : {len(val_samples)} samples (20%)")

train_dataset = CachedSorghumDataset(train_samples, is_train=True)
val_dataset = CachedSorghumDataset(val_samples, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, pin_memory=True)

# ==========================================
# 3. SWIN TRANSFORMER MODEL ARCHITECTURE
# ==========================================
print("\nInitializing Swin Transformer Model (swin_tiny_patch4_window7_224)...", flush=True)
model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True, num_classes=len(classes))
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
scaler = torch.cuda.amp.GradScaler()  # Automatic Mixed Precision for Tensor Cores

epochs = 20
history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
total_batches = len(train_loader)

# ==========================================
# 4. TRAINING WITH REAL-TIME PROGRESS
# ==========================================
print(f"\n--- Starting 20 Epochs GPU Training on {device} ---", flush=True)
start_time = time.time()

for epoch in range(epochs):
    ep_start = time.time()
    model.train()
    t_loss, t_corr = 0.0, 0
    
    for step, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        t_loss += loss.item() * inputs.size(0)
        t_corr += torch.sum(outputs.argmax(1) == labels)

        if (step + 1) % 30 == 0 or (step + 1) == total_batches:
            pct = ((step + 1) / total_batches) * 100
            current_acc = (t_corr.item() / ((step + 1) * train_loader.batch_size)) * 100
            print(f"Epoch [{epoch+1:02d}/{epochs:02d}] Batch {step+1:03d}/{total_batches:03d} ({pct:5.1f}%) | Current Acc: {current_acc:.2f}% | Loss: {loss.item():.4f}", flush=True)

    model.eval()
    v_loss, v_corr = 0.0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            v_loss += loss.item() * inputs.size(0)
            v_corr += torch.sum(outputs.argmax(1) == labels)

    tr_acc = float(t_corr.double().item() / len(train_dataset))
    vl_acc = float(v_corr.double().item() / len(val_dataset))
    tr_loss = float(t_loss / len(train_dataset))
    vl_loss = float(v_loss / len(val_dataset))

    history['train_loss'].append(tr_loss)
    history['val_loss'].append(vl_loss)
    history['train_acc'].append(tr_acc)
    history['val_acc'].append(vl_acc)

    ep_time = time.time() - ep_start
    print(f"==> Epoch [{epoch+1:02d}/{epochs:02d}] Finished in {ep_time:.2f}s | Train Loss: {tr_loss:.4f} Acc: {tr_acc:.4f} | Val Loss: {vl_loss:.4f} Acc: {vl_acc:.4f}\n", flush=True)

total_time = time.time() - start_time
print(f"Training Complete in {total_time/60:.2f} minutes!", flush=True)

# ==========================================
# 5. SAVE METRICS & PLOT CHARTS
# ==========================================
print("\nEvaluating final validation metrics & exporting files...", flush=True)
model.eval()
preds, targets, probs = [], [], []
with torch.no_grad():
    for inputs, labels in val_loader:
        inputs = inputs.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
        probs.extend(F.softmax(outputs, dim=1).cpu().numpy())
        preds.extend(outputs.argmax(1).cpu().numpy())
        targets.extend(labels.numpy())

preds, targets, probs = np.array(preds), np.array(targets), np.array(probs)

acc = accuracy_score(targets, preds)
precision, recall, f1, _ = precision_recall_fscore_support(targets, preds, average='weighted')
cm = confusion_matrix(targets, preds)
cls_report = classification_report(targets, preds, target_names=classes, output_dict=True)

json_results = {
    "model": "Swin Transformer GPU (Swin-Tiny)",
    "epochs": epochs,
    "classes": classes,
    "overall_metrics": {
        "accuracy": float(acc),
        "weighted_precision": float(precision),
        "weighted_recall": float(recall),
        "weighted_f1_score": float(f1)
    },
    "training_history": history,
    "confusion_matrix": cm.tolist(),
    "classification_report": cls_report
}

with open("swin_training_metrics.json", "w") as f:
    json.dump(json_results, f, indent=4)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
epoch_range = range(1, epochs + 1)

axes[0, 0].plot(epoch_range, history['train_acc'], 'b-o', label='Train Accuracy')
axes[0, 0].plot(epoch_range, history['val_acc'], 'r-o', label='Val Accuracy')
axes[0, 0].set_title('Swin Transformer: Accuracy over Epochs')
axes[0, 0].legend()
axes[0, 0].grid(True)

axes[0, 1].plot(epoch_range, history['train_loss'], 'b-o', label='Train Loss')
axes[0, 1].plot(epoch_range, history['val_loss'], 'r-o', label='Val Loss')
axes[0, 1].set_title('Swin Transformer: Loss over Epochs')
axes[0, 1].legend()
axes[0, 1].grid(True)

sns.heatmap(cm, annot=True, fmt='d', xticklabels=classes, yticklabels=classes, ax=axes[1, 0], cmap='Blues')
axes[1, 0].set_title('Swin Transformer: Confusion Matrix')

bin_targets = label_binarize(targets, classes=list(range(len(classes))))
for i, cls in enumerate(classes):
    fpr, tpr, _ = roc_curve(bin_targets[:, i], probs[:, i])
    axes[1, 1].plot(fpr, tpr, label=f'{cls} (AUC = {auc(fpr, tpr):.2f})')
axes[1, 1].plot([0, 1], [0, 1], 'k--')
axes[1, 1].set_title('Swin Transformer: ROC Curves by Class')
axes[1, 1].legend(loc='lower right')
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('swin_transformer_20epochs_results.png')
torch.save(model.state_dict(), "swin_transformer_20epochs.pth")

print("\nSUCCESS! Swin Transformer output files saved:")
print(" -> swin_training_metrics.json")
print(" -> swin_transformer_20epochs_results.png")
print(" -> swin_transformer_20epochs.pth")
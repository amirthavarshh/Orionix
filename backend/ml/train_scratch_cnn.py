import os
import time
import json
import csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model Definition
class EuroSATCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(EuroSATCNN, self).__init__()
        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4: 128 -> 256
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def main():
    data_dir = "backend/ml/data/eurosat"
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")
    
    # Ensure ml directory exists
    os.makedirs("backend/ml", exist_ok=True)
    
    # Check if data exists
    if not os.path.exists(train_dir):
        print(f"Error: Training directory '{train_dir}' not found. Did you run download_eurosat.py?")
        return

    print(f"Using device: {device}")

    # Data Transforms
    # For EuroSAT, images are 64x64
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # Standard ImageNet normalization values
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    transform_val_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print("Loading datasets...")
    train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
    val_dataset = datasets.ImageFolder(val_dir, transform=transform_val_test)
    test_dataset = datasets.ImageFolder(test_dir, transform=transform_val_test)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print(f"Classes identified ({len(class_names)}): {class_names}")
    
    # Initialize Model, Loss, and Optimizer
    model = EuroSATCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training Configuration
    epochs = 30
    patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = []
    
    print("\nStarting training phase...")
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training Phase
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total
        
        # Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        
        elapsed_time = time.time() - start_time
        
        print(f"Epoch [{epoch+1:02d}/{epochs}] ({elapsed_time:.1f}s) "
              f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f} | "
              f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}")
              
        history.append({
            "epoch": epoch + 1,
            "train_loss": round(epoch_train_loss, 4),
            "train_acc": round(epoch_train_acc, 4),
            "val_loss": round(epoch_val_loss, 4),
            "val_acc": round(epoch_val_acc, 4)
        })
        
        # Early Stopping & Model Checkpointing
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), "backend/ml/eurosat_scratch_cnn.pt")
            print(f"  => Validation loss decreased to {best_val_loss:.4f}. Saved best model.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs (No improvement in {patience} epochs).")
                break

    # Save Training History
    csv_path = "backend/ml/training_history.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])
        writer.writeheader()
        writer.writerows(history)
    print(f"\nTraining history logged to {csv_path}")

    # Load best model for testing
    print("\nLoading best model weights for evaluation...")
    model.load_state_dict(torch.load("backend/ml/eurosat_scratch_cnn.pt"))
    model.eval()
    
    # Testing Phase
    print("Evaluating on held-out test set...")
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # Calculate Evaluation Metrics
    test_acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=range(len(class_names)))
    
    class_metrics = {}
    for i, cls in enumerate(class_names):
        class_metrics[cls] = {
            "precision": float(round(precision[i], 4)),
            "recall": float(round(recall[i], 4)),
            "f1-score": float(round(f1[i], 4)),
            "support": int(support[i])
        }
        
    results = {
        "overall_test_accuracy": float(round(test_acc, 4)),
        "per_class_metrics": class_metrics
    }
    
    with open("backend/ml/scratch_results.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"Evaluation metrics saved to backend/ml/scratch_results.json")
    print(f"Overall Test Accuracy: {test_acc * 100:.2f}%")
    
    # Generate and Save Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('EuroSAT Test Set Confusion Matrix - Scratch CNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("backend/ml/confusion_matrix.png")
    print("Confusion matrix saved to backend/ml/confusion_matrix.png")

if __name__ == "__main__":
    main()

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.hybrid_model import HybridDeepfakeDetector, SimpleCNNModel

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, transform=None, sequence_length=4):
        self.root_dir = root_dir
        self.transform = transform
        self.sequence_length = sequence_length
        
        self.real_dir = os.path.join(root_dir, 'real')
        self.fake_dir = os.path.join(root_dir, 'fake')
        
        self.real_images = [os.path.join(self.real_dir, f) for f in os.listdir(self.real_dir) if f.endswith('.jpg')]
        self.fake_images = [os.path.join(self.fake_dir, f) for f in os.listdir(self.fake_dir) if f.endswith('.jpg')]
        
        self.all_images = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)
        
        # Group images by video
        self.video_groups = {}
        for img_path, label in zip(self.all_images, self.labels):
            video_name = '_'.join(os.path.basename(img_path).split('_')[:-1])
            if video_name not in self.video_groups:
                self.video_groups[video_name] = {'images': [], 'label': label}
            self.video_groups[video_name]['images'].append(img_path)
        
        # Create sequences
        self.sequences = []
        for video_name, data in self.video_groups.items():
            images = sorted(data['images'])
            label = data['label']
            
            # Create sequences with larger step to reduce total sequences
            for i in range(0, len(images) - self.sequence_length + 1, self.sequence_length):
                sequence = images[i:i + self.sequence_length]
                self.sequences.append({'images': sequence, 'label': label, 'length': len(sequence)})

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence_data = self.sequences[idx]
        images = []
        
        for img_path in sequence_data['images']:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)
        
        # Pad sequence if necessary
        while len(images) < self.sequence_length:
            images.append(torch.zeros_like(images[0]))
        
        images = torch.stack(images)
        label = torch.tensor(sequence_data['label'], dtype=torch.float32)
        length = torch.tensor(sequence_data['length'], dtype=torch.long)
        
        return images, label, length

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def train_model():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    base_dir = r"C:\Users\Sowmya\Downloads\Deep_fake"
    train_dir = os.path.join(base_dir, "dataset", "train")
    val_dir = os.path.join(base_dir, "dataset", "val")
    
    # Create datasets with smaller sequence length
    train_dataset = DeepfakeDataset(train_dir, transform=train_transform, sequence_length=4)
    val_dataset = DeepfakeDataset(val_dir, transform=val_transform, sequence_length=4)
    
    # Smaller batch size
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # Initialize model
    print("Using SimpleCNNModel")
    model = SimpleCNNModel().to(device)
    
    # Use BCEWithLogitsLoss
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)
    
    # Training loop
    num_epochs = 10
    best_val_loss = float('inf')
    
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, (images, labels, lengths) in enumerate(train_pbar):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Fix shape issue: ensure outputs and labels have same shape
            outputs = outputs.view(-1)  # Flatten to [batch_size]
            
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Apply sigmoid for accuracy calculation
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            # Clear memory
            torch.cuda.empty_cache()
            
            train_pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{correct/total:.3f}'
            })
        
        train_loss = running_loss / len(train_loader)
        train_accuracy = correct / total
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for images, labels, lengths in val_pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                outputs = outputs.view(-1)  # Flatten to [batch_size]
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                # Apply sigmoid for accuracy calculation
                probabilities = torch.sigmoid(outputs)
                predicted = (probabilities > 0.5).float()
                val_correct += (predicted == labels).sum().item()
                val_total += labels.size(0)
                
                all_preds.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                val_pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{val_correct/val_total:.3f}'
                })
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        # Calculate additional metrics
        val_precision = precision_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds], zero_division=0)
        val_recall = recall_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds], zero_division=0)
        val_f1 = f1_score(all_labels, [1 if p > 0.5 else 0 for p in all_preds], zero_division=0)
        val_auc = roc_auc_score(all_labels, all_preds) if len(np.unique(all_labels)) > 1 else 0.5
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')
        print(f'Val Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
            print('Saved best model!')
        
        scheduler.step(val_loss)
        torch.cuda.empty_cache()
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.savefig(os.path.join(base_dir, 'training_history.png'))
    plt.show()

if __name__ == "__main__":
    train_model()
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
from imblearn.over_sampling import SMOTE
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision.models import resnet18,resnet50, ResNet18_Weights,ResNet34_Weights, VGG16_Weights, resnet50, ResNet50_Weights, ResNet101_Weights

# Load the image and label arrays
image_array = np.load('Image_array224x224.npy')
image_labels = np.load('label_array224x224.npy')

# Target labels are decremented by 1 to ensure zero-based indexing
target = image_labels.astype(int) - 1

# Hyperparameters
num_classes = 5
num_epochs = 10
learning_rate = 1e-4
optimizer_type = 'adam' # Or SGD
momentum = 0.9 # For SGD
weight_decay = 1e-5 # For SGD
dropout_prob = 0.5
sch_step_size = 80
sch_gamma = 0.1
scheduler_on = True
batch_size_list = [64]
unfreeze_list = [3, 4]  # Number of Layers to unfreeze
dropout_prob_list = [0.4, 0.3]
resnet_version_list = [34, 50, 101]
oversample = True   # This will use SMOTE


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset class for loading data
class BuildingDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)  # Ensure image data is in uint8
        image = Image.fromarray(image)  # Convert NumPy array to PIL image
        
        if self.transform:
            image = self.transform(image)
        
        return image, self.labels[idx]

# Training function
def train_model():
    train_acc, val_acc = [], []
    
    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Calculate training accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_acc.append(100 * correct_train / total_train)

        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        val_acc.append(100 * correct_val / total_val)

        # Update learning rate if scheduler is used
        if scheduler_on:
            scheduler.step()
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Accuracy: {train_acc[-1]:.2f}%, Validation Accuracy: {val_acc[-1]:.2f}%, LR: {scheduler.get_lr()}')
        else:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Accuracy: {train_acc[-1]:.2f}%, Validation Accuracy: {val_acc[-1]:.2f}%')

        # Save model checkpoint periodically
        if epoch % 10 == 0:
            val_accuracy = 100 * correct_val / total_val
            torch.save(model, f'##Pytorch_{remarks}_val{val_accuracy}%_ep{epoch}_checkpoint.pth')
            print(f'Checkpoint Saved: dropout = {dropout_prob}, unfreeze layers = {layers_unfreeze}')

    return train_acc, val_acc

# Main training loop with different configurations
for resnet_version in resnet_version_list:
    remarks = f'Test_run_Train_ResNET{resnet_version}_scheduler{scheduler_on}_SMOTE{oversample}_optim{optimizer_type}'

    class BuildingClassifierWithDropout(nn.Module):
        def __init__(self, num_classes=5, dropout_prob=0.5, unfreeze_layers=1):
            super(BuildingClassifierWithDropout, self).__init__()
            # Using pretrained ResNet101 for transfer learning
            if(resnet_version == 101):
                self.model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V2)
            elif(resnet_version == 18):
                self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
            elif(resnet_version == 34):
                self.model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
            elif(resnet_version == 50):
                self.model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)

            # Freeze all layers first
            for param in self.model.parameters():
                param.requires_grad = False

            # Option to unfreeze more layers
            if unfreeze_layers == 1:
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif unfreeze_layers == 2:
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            elif unfreeze_layers == 3:
                for param in self.model.layer3.parameters():
                    param.requires_grad = True
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
                for param in self.model.fc.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.parameters():
                    param.requires_grad = True

            
            self.dropout_fc = nn.Dropout(dropout_prob)    # Dropout before the final fully connected layer
            
            # Modify the final fully connected layer
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        
        
        def forward(self, x):
            # Pass through the ResNet18 feature extractor
            x = self.model(x)
            # Apply dropout before the final layer
            x = self.dropout_fc(x)
            return x
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB
        transforms.Resize((224, 224)), # Add random rotation + horizontal flipping
        # transforms.RandomHorizontalFlip(p=0.5),       # Add random horizontal flip
        # transforms.RandomRotation(degrees=30),
        transforms.ToTensor(),
        # transforms.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0))
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])
    
    for batch_size in batch_size_list:
        if oversample:
            # Apply SMOTE for handling class imbalance
            X_train, X_val, y_train, y_val = train_test_split(image_array, target, train_size=0.8, stratify=target, random_state=42)
            X = X_train.reshape((X_train.shape[0], -1))
            smote = SMOTE(sampling_strategy='auto', random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y_train)
            X_resampled = X_resampled.reshape((X_resampled.shape[0], X_train.shape[1], X_train.shape[2]))
            train_dataset = BuildingDataset(X_resampled, y_resampled, transform=transform)
            val_dataset = BuildingDataset(X_val, y_val, transform=transform)
        else:
            train_dataset = BuildingDataset(X_train, y_train, transform=transform)
            val_dataset = BuildingDataset(X_val, y_val, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        for dropout_prob in dropout_prob_list:
            for layers_unfreeze in unfreeze_list:
                # Model initialization
                model = BuildingClassifierWithDropout(num_classes=num_classes, dropout_prob=dropout_prob, unfreeze_layers=layers_unfreeze).to(device)

                # Loss and optimizer
                criterion = nn.CrossEntropyLoss()
                if optimizer_type == 'sgd':
                    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
                else:
                    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
                
                # Learning rate scheduler
                if scheduler_on:
                    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=sch_step_size, gamma=sch_gamma)

                # Train the model
                train_acc, val_acc = train_model()

                # Save the trained model
                torch.save(model.state_dict(), f'###Pytorch_{remarks}_ep{num_epochs}_batch{batch_size}.pth')
                
                # Plot accuracy curves
                plt.figure(figsize=(10, 5))
                plt.plot(range(num_epochs), train_acc, label="Train Accuracy")
                plt.plot(range(num_epochs), val_acc, label="Validation Accuracy")
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy (%)')
                plt.title('Train vs Validation Accuracy')
                plt.legend()
                plt.savefig(f'##{remarks}_batch{batch_size}.jpg')
                plt.close()

                # Confusion Matrix calculation and plotting
                def compute_confusion_matrix(model, val_loader):
                    all_preds, all_labels = [], []
                    model.eval()
                    with torch.no_grad():
                        for images, labels in val_loader:
                            images, labels = images.to(device), labels.to(device)
                            outputs = model(images)
                            _, preds = torch.max(outputs, 1)
                            all_preds.extend(preds.cpu().numpy())
                            all_labels.extend(labels.cpu().numpy())
                    cm = confusion_matrix(all_labels, all_preds)
                    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                    return cm_normalized

                cm_normalized = compute_confusion_matrix(model, val_loader)
                sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues')
                plt.title(f'Normalized Confusion Matrix for {remarks}')
                plt.savefig(f'#Confusion_Matrix_{remarks}.jpg')
                plt.close()

                # Clean up and empty GPU memory
                del model
                torch.cuda.empty_cache()

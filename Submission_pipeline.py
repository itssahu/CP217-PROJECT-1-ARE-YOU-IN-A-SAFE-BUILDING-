import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torchvision.models as models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns
import pandas as pd
import torch
import numpy as np
import torch.nn.functional as F
from collections import Counter
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np
from collections import Counter


# Path to Test_Data
path_test = 'Test_Data'

Submission_filename = 'Submission_19_test_1.csv'

######################### This Part is for the pretrained models to load ##################

class BuildingDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

    ### comment this part out for monochrome input
        image = image.astype(np.uint8)

        # Convert NumPy array to PIL image
        image = Image.fromarray(image)
    #################################################

        if self.transform:
            image = self.transform(image)
            
        return image, label



# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define CNN model (Transfer Learning with ResNet18)

class BuildingClassifierWithDropout(nn.Module):
    def __init__(self, num_classes=5, dropout_prob=0.5, unfreeze_layers=1,resnet_version=50):
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

        # Dropout layers
        # self.dropout_conv = nn.Dropout(dropout_prob)  # Dropout for convolutional layers
        self.dropout_fc = nn.Dropout(dropout_prob)    # Dropout before the final fully connected layer
        
        # Modify the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
            # Pass through the ResNet18 feature extractor
            x = self.model(x)
            # Apply dropout before the final layer
            x = self.dropout_fc(x)
            return x


class BuildingClassifier(nn.Module):
    def __init__(self, num_classes=5, dropout_prob=0.5):
        super(BuildingClassifier, self).__init__()
        # Using pretrained ResNet18 for transfer learning
        self.model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        # Dropout layer
        self.dropout = nn.Dropout(dropout_prob)
        # Modify the final fully connected layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    
    def forward(self, x):
        # Pass through the ResNet18 feature extractor
        x = self.model(x)
        # Apply dropout before the final layer
        x = self.dropout(x)
        return x
    

# Function to load models and apply max voting ensemble
def max_voting_ensemble(model_names_list, val_loader, num_classes, device):
    all_model_preds = []

    # Iterate through each model in the model_names_list
    for model_name in model_names_list:
        print(f"Loading model: {model_name}")

        if('ResNET18' in model_name):
            resnet_version = 18
        elif('ResNET34' in model_name):
            resnet_version = 34
        elif('ResNET50' in model_name):
            resnet_version = 50
        elif('ResNET101' in model_name):
            resnet_version = 101

        # Load the saved model
        # model = BuildingClassifierWithDropout(num_classes=num_classes, dropout_prob=0.5, unfreeze_layers=2,resnet_version=resnet_version)  # Customize based on your model structure
        # model.load_state_dict(torch.load(model_name, map_location=device))
        model = torch.load(model_name, map_location=device)
        model = model.to(device)
        model.eval()  # Set model to evaluation mode

        # Collect predictions for this model
        model_preds = []
        with torch.no_grad():
            for images, _ in val_loader:  # We don't need the labels for prediction
                images = images.to(device)

                # Make predictions
                outputs = model(images)
                _, preds = torch.max(outputs, 1)

                # Append predictions to the model_preds list
                model_preds.extend(preds.cpu().numpy())

        all_model_preds.append(model_preds)

    # Transpose to get predictions for each sample across all models
    all_model_preds = np.array(all_model_preds).T  # Shape: [num_samples, num_models]

    # Max voting: Get the most common class for each sample
    final_predictions = []
    for sample_preds in all_model_preds:
        # Count the occurrence of each class
        most_common_class = Counter(sample_preds).most_common(1)[0][0]
        final_predictions.append(most_common_class)

    return np.array(final_predictions)




# Function to load models and apply softmax-based ensemble
def softmax_ensemble(model_names_list, val_loader, num_classes, device):
    all_model_softmax = []

    # Iterate through each model in the model_names_list
    for model_name in model_names_list:
        print(f"Loading model: {model_name}")

        if 'ResNET18' in model_name:
            resnet_version = 18
        elif 'ResNET34' in model_name:
            resnet_version = 34
        elif 'ResNET50' in model_name:
            resnet_version = 50
        elif 'ResNET101' in model_name:
            resnet_version = 101

        # Load the saved model
        model = torch.load(model_name, map_location=device)
        model = model.to(device)
        model.eval()  # Set model to evaluation mode

        # Collect softmax probabilities for this model
        model_softmax = []
        with torch.no_grad():
            for images, _ in val_loader:  # We don't need the labels for prediction
                images = images.to(device)

                # Make predictions
                outputs = model(images)

                # Apply softmax to get probabilities
                softmax_outputs = F.softmax(outputs, dim=1)

                # Append softmax probabilities to the model_softmax list
                model_softmax.append(softmax_outputs.cpu().numpy())

        all_model_softmax.append(np.concatenate(model_softmax, axis=0))  # Shape: [num_samples, num_classes]

    # Stack the softmax outputs from all models: Shape [num_models, num_samples, num_classes]
    all_model_softmax = np.array(all_model_softmax)

    # Take the mean softmax output across all models: Shape [num_samples, num_classes]
    mean_softmax = np.mean(all_model_softmax, axis=0)

    # Choose the class with the highest averaged probability
    final_predictions = np.argmax(mean_softmax, axis=1)

    return final_predictions

################################################################################


filenames_ = [filen for filen in os.listdir(f"{path_test}/") if 'jpg' in filen]
print(len(filenames_))

filenames = [f'{i+1}.jpg' for i in range(len(filenames_))]
print(filenames)



def store_img_return_arr(filenames_list,path_to):
    store_names = np.zeros((len(filenames_list),224,224)) # num_of_files x 224 x 224, initialization
    
    for i in range(len(filenames_list)): # Loop through all files in that folder--> store as np array
        store_names[i,:,:] = np.asarray((Image.open(f'{path_to}/{filenames_list[i]}')).resize((224, 224)))
    return store_names

store_test = store_img_return_arr(filenames, path_test)




# Assume store_test has shape (num_of_images, 224, 224)
# Convert store_test array to a 3-channel format using transformations in DataLoader

class StoreTestDataset(Dataset):
    def __init__(self, store_test, transform=None):
        self.store_test = store_test
        self.transform = transform
        self.dummy_label = -1  # Dummy label for test data (as you don't have actual labels)

    def __len__(self):
        return len(self.store_test)

    def __getitem__(self, idx):
        # Extract the single-channel image and convert it to PIL Image
        img = Image.fromarray(self.store_test[idx].astype(np.uint8))  # Convert to PIL Image for transforms
        
        # Apply transformations (Grayscale to RGB and other transforms)
        if self.transform:
            img = self.transform(img)
        
        return img, self.dummy_label  # Return both the image and the dummy label

# Define the same transform you used for the validation set
transform_test = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),  # Convert grayscale to 3-channel RGB for the model
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalizing same as ImageNet
])

# Create the test dataset from the store_test array
test_dataset = StoreTestDataset(store_test, transform=transform_test)

# Create the DataLoader for test images (with dummy labels)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

# Checking the size of the dataset and test_loader
print(f"Test dataset length: {len(test_dataset)}")
print(f"Test loader length: {len(test_loader)} batches")



Model_names_19 = ['FULL_Pytorch_ep100_batch64_lr0.0001_drop0.4_lr0.0001_sub_5.pth',
'Pytorch_Invest_SMOTE_ResNET50_schedulerTrue_val63.29365079365079%_ep50_checkpoint_0.6_batch64.pth', 
'Pytorch_Multiple_layers_ResNET50_schedulerTrue_val64.88095238095238%_ep170_checkpoint_0.4_unfrozen4_batch64.pth', 
'Pytorch_Multiple_layers_ResNET50_schedulerTrue_val65.47619047619048%_ep130_checkpoint_0.4_unfrozen3_batch64.pth', 
'FULL_Pytorch_cleaned_ResNET50_schedulerTrue_ep260_batch64_lr0.0001_drop0.5_unfrozen1_lr0.0001.pth', 
'FULL_Pytorch_FineTune_Train_ResNET50_schedulerTrue_SMOTEFalse_optimadam_ep150_batch64_lr0.0001_drop0.5_unfrozen1_lr0.0001.pth']

resnet_preds = max_voting_ensemble(model_names_list=Model_names_19,val_loader=test_loader,num_classes=5,device=device)

resnet_preds_test = np.array(resnet_preds)+1

d = {'ID': np.array([i+1 for i in range(len(filenames))]), 'Predictions': resnet_preds_test}
df = pd.DataFrame(data=d)

df.to_csv(f'{Submission_filename}',index=False)
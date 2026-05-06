import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
import os
import time

# Load the scr modules.
from src.dataset import PigPostureDataset
from src.model import PigPosture_CNN
from src.train import train_model
from src.visualize import visualize_acc_loss, visualize_errors
'''
https://stackoverflow.com/questions/58151507/why-pytorch-officially-use-mean-0-485-0-456-0-406-and-std-0-229-0-224-0-2
If you want to train from scratch on your own dataset, you can calculate the new mean and std. 
Otherwise, using the Imagenet pretrianed model with its own mean and std is recommended.
Choice - But I didn't have enough time to calculate the mean and std.
I choose the values from the defualt setting of ImageNet.
'''
print("--- [START] Execution Pipeline Initiated ---")
print("[1/7] Setting up Image Transformations...")
T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),                #
    transforms.Normalize(                 #
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
print(f"Normalization: ImageNet Defaults")
print(f"Resize Dimensions: 224x224")


# From the dataset.py
print("[2/7] Loading PigPostureDataset...")
dataset = PigPostureDataset(
    csv_file='data/csv/train.csv', img_dir='./data/images/train_processed_images', transform=T)
total_count = len(dataset)
print(f"Total images found in CSV: {total_count}")

# 3. Data Splitting
print("[3/7] Splitting dataset into Train (80%) and Validation (20%)...")
train_size = int(len(dataset) * 0.8)
val_size = len(dataset) - train_size
gen = torch.Generator().manual_seed(1)  # For reproducability.
trainset, valset = random_split(
    # if we set the generator = gen same value, it ensure the train and validate images are same all the time.
    dataset, [train_size, val_size], generator=gen)

print("[4/7] Initializing DataLoaders...")
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
# We don't have to update the weights so choose the batch_size = 512.
val_loader = DataLoader(valset, batch_size=512)

print(f"Train Loader: {len(train_loader)} batches (BS=64)")
print(f"Val   Loader: {len(val_loader)} batches (BS=512)")

# Build and train the model.
print("[5/7] Building PigPosture_CNN Architecture...")
model = PigPosture_CNN()

print("[6/7] Starting Training Loop...")
print("--------------------------------------------------")
start_time = time.time()
results = train_model(model, train_loader, val_loader,
                      patience=10, min_epochs=10, max_epochs=100)
end_time = time.time()
elapsed = end_time - start_time

hours = int(elapsed // 3600)
minutes = int((elapsed % 3600) // 60)
seconds = elapsed % 60

print(f"Training time: {hours}h {minutes}m {seconds:.2f}s")

# Save Model's weights
outputs_dir = '/content/drive/MyDrive/Pig_Project/pig_posture_GPU/outputs'
os.makedirs(outputs_dir, exist_ok=True)
save_path = os.path.join(outputs_dir, 'best_pig_model.pth')

print(f"[7/7] Saving Model Weights to {save_path}...")
torch.save(model.state_dict(), save_path)

print("--- [FINISH] Pipeline Completed Successfully ---")

print("--- Save The Analysis---")
visualize_acc_loss(results, val_size)
visualize_errors(model, val_loader)

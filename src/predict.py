import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.model import PigPosture_CNN, device


class PigPostureTestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row_id = self.annotations.iloc[idx]["row_id"]
        img_path = os.path.join(self.img_dir, row_id + ".jpg")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image


T = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_dataset = PigPostureTestDataset(
    csv_file="data/csv/sample_submission.csv",
    img_dir="data/images/test_processed_images",
    transform=T
)

test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

model = PigPosture_CNN()

model_path = "/content/drive/MyDrive/Pig_Project/pig_posture_GPU/outputs/best_pig_model.pth"
model.load_state_dict(torch.load(model_path, map_location=device))

model.to(device)
model.eval()


predictions = []

with torch.no_grad():
    for data in test_loader:
        data = data.to(device)

        output = model(data)
        pred = output.argmax(dim=1)

        predictions.extend(pred.cpu().numpy())

submission = pd.read_csv("data/csv/sample_submission.csv")
submission["class_id"] = predictions

output_dir = "/content/drive/MyDrive/Pig_Project/pig_posture_GPU/outputs"
os.makedirs(output_dir, exist_ok=True)

submission_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(submission_path, index=False)


print("Saved submission to outputs/submission.csv")

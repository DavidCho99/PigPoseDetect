import pandas as pd
import cv2
import ast
import os
# https://opencv.org/cropping-an-image-using-opencv/

df = pd.read_csv('train.csv')

# Add column for processed_image
df["processed"] = df['row_id'] + '.jpg'

image_dir = 'data/train_images'
saved_folder = 'data/train_processed_images'

os.makedirs(saved_folder, exist_ok=True)

for i, row in df.iterrows():
    print(row["row_id"])
    # 1. Create the image path
    img_path = os.path.join(image_dir, row["image_id"])
    # 2. Read image from the path
    img = cv2.imread(img_path)

    if img is None:
        continue

    # 3. Read bbox from the csv (Convert it to list using ast.literal_eval)
    bbox = ast.literal_eval(row['bbox'])
    x, y, w, h = map(int, bbox)

    # 4. Crop it
    cropped = img[y:y+h, x:x+w]

    # 5. Save the name with row_id + jpg under processed_images
    save_path = os.path.join("data/processed_images", row['row_id'] + '.jpg')
    cv2.imwrite(save_path, cropped)

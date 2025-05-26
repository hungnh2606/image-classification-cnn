import os
import shutil
from sklearn.model_selection import train_test_split

# Đường dẫn thư mục chứa ảnh
image_dir = "D:\Image classification\data\dataset"
output_dir = "D:\Image classification\data\dataset"

images = []
labels = []

for class_name in os.listdir(image_dir):
    class_path = os.path.join(image_dir, class_name)
    print(class_path)
    if os.path.isdir(class_path):
        for img_name in os.listdir(class_path):
            images.append(os.path.join(class_path, img_name))
            labels.append(class_name)

print(len(images))
print(len(labels))

X_train, X_temp, y_train, y_temp = train_test_split(images, labels, test_size=0.3, random_state=42)  # 60% train, 40% temp

X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 20% val, 20% test

print(f"Train: {len(X_train)} mẫu")
print(f"Validation: {len(X_val)} mẫu")
print(f"Test: {len(X_test)} mẫu")

def save_images(image_list, label_list, folder):
    for img_path, label in zip(image_list, label_list):
        dest_folder = os.path.join(output_dir, folder, label)
        os.makedirs(dest_folder, exist_ok=True)
        shutil.copy(img_path, dest_folder)

# Lưu ảnh vào thư mục train và test
save_images(X_train, y_train, "Train")
save_images(X_val, y_val, "Validate")
save_images(X_test, y_test, "Test")
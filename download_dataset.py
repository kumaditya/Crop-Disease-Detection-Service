from datasets import load_dataset
import os

print("Loading dataset...")

dataset = load_dataset(
    "Project-AgML/tomato_leaf_disease",
    split="train"
)

print("Total images:", len(dataset))

# Dataset ke actual label names
label_names = dataset.features["label"].names

print("\nAvailable classes:")
for i, name in enumerate(label_names):
    print(i, "->", name)


# Hume sirf ye 3 classes chahiye
selected_classes = {
    "Early Blight": "Tomato___Early_blight",
    "Late Blight": "Tomato___Late_blight",
    "Healthy": "Tomato___healthy"
}

output_dir = "data/dataset"

for folder in selected_classes.values():
    os.makedirs(
        os.path.join(output_dir, folder),
        exist_ok=True
    )


count = {
    "Early Blight": 0,
    "Late Blight": 0,
    "Healthy": 0
}


print("\nSaving selected images...")

for i, item in enumerate(dataset):

    # label is integer
    label_id = item["label"]

    # integer -> actual label name
    label_name = label_names[label_id]

    if label_name not in selected_classes:
        continue

    folder_name = selected_classes[label_name]

    image = item["image"]

    filename = f"{label_name.replace(' ', '_')}_{i}.jpg"

    path = os.path.join(
        output_dir,
        folder_name,
        filename
    )

    image.save(path)

    count[label_name] += 1


print("\n==============================")
print("Download completed!")
print("==============================")

print("Early Blight:", count["Early Blight"])
print("Late Blight :", count["Late Blight"])
print("Healthy     :", count["Healthy"])
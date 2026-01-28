from dataset_loader import TrueLensDataset

dataset = TrueLensDataset("dataset")
print("Total images:", len(dataset))

img, label = dataset[0]
print("Image shape:", img.shape)
print("Label:", label)

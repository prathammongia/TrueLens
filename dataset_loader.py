import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class TrueLensDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = ["real", "fake"]
        self.image_paths = []
        self.labels = []

        for label, cls in enumerate(self.classes):
            cls_path = os.path.join(root_dir, cls)
            for img in os.listdir(cls_path):
                if img.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(cls_path, img))
                    self.labels.append(label)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(image)
        label = self.labels[idx]
        return image, label

from dataset_loader import TrueLensDataset
from fft_utils import fft_transform

class TrueLensFFTDataset(TrueLensDataset):
    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        fft_img = fft_transform(image)
        return image, fft_img, label

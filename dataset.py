from torch.utils.data import Dataset
import PIL.Image as Image
import os

class FaceDataset(Dataset):
    def __init__(self, root, transform = None):
        super(FaceDataset, self).__init__()
        self.imgs = [root + '/' + i for i in os.listdir(root)]
        self.transform = transform
        print("Number of data:", len(self.imgs))

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img = Image.open(img_path)
        img = self.transform(img)
        return img

    def __len__(self):
        return len(self.imgs)
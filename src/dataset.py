import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class PlantSeedDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        PlantSeedDataset
        :param data_dir: str, 数据集所在路径
        :param transform: torch.transform, 数据预处理
        """  
        self.transform = transform
        self.name_dic = {'Black-grass': 0, 'Charlock': 1, 'Cleavers': 2, 'Common Chickweed': 3, 
                'Common wheat': 4, 'Fat Hen': 5, 'Loose Silky-bent': 6, 'Maize': 7,
                'Scentless Mayweed': 8, 'Shepherds Purse': 9, 'Small-flowered Cranesbill': 10, 'Sugar beet': 11}
        self.path_imgs, self.labels, self.imgs_name, self.data_info = self.get_img_info(data_dir)
 
    def __getitem__(self, index):
        path_img, label, img_name = self.data_info[index]
        img = Image.open(path_img).convert('RGB')     
 
        if self.transform is not None:
            img = self.transform(img)   
        return img, label, img_name
 
    def __len__(self):
        return len(self.data_info)
 
    # @staticmethod
    def get_img_info(self, data_dir):
        data_info = []
        self.imgs = []
        path_imgs = []
        labels = []
        imgs_name = []        
        for root, dirs, _ in os.walk(data_dir):
            for sub_dir in dirs:
                img_names = os.listdir(os.path.join(root, sub_dir))
                img_names = list(filter(lambda x: x.endswith('.png'), img_names))
                # img_count = len(img_names)
                # print(len(img_names))
                for i in range(len(img_names)):
                    img_name = img_names[i]
                    path_img = os.path.join(root, sub_dir, img_name)
                    label = sub_dir
                    path_imgs.append(path_img)
                    img = Image.open(path_img).convert('RGB')     
                    if self.transform is not None:
                        img = self.transform(img)
                    self.imgs.append(img)
                    labels.append(self.name_dic[label])
                    
                    data_info.append((path_img, label, img_name))
                imgs_name.append(img_names)
 
        return path_imgs, labels, imgs_name, data_info
import os
import PIL

from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets import VisionDataset


# class FFHQ(VisionDataset):
#     """ 
#     Modified CelebA dataset to adapt for custom cropped images.
#     """

#     def __init__(
#             self,
#             root='/home/comp/f1251215/Re-thinking_MI/datasets/ffhq/thumbnails128x128',
#             transform: Optional[Callable] = None,
#     ):
#         super(FFHQ, self).__init__(root, transform=transform)
#         self.filename = os.listdir(root)[:20000]

#     def __len__(self):
#         return len(self.filename)
    
    
#     def __getitem__(self, index):
#         file_path = os.path.join(self.root, self.filename[index])

#         if os.path.exists(file_path) == False:
#             file_path = file_path.replace('.jpg', '.png')
#         im = PIL.Image.open(file_path)

#         if self.transform:
#             return self.transform(im)
#         else:
#             return im



class FFHQ(VisionDataset):
    """ 
    FFHQ dataset loader adapted to include images from nested directories.
    Each of the 70 directories contains 1000 images.
    """
    def __init__(
        self,
        root: str = '/home/comp/f1251215/Re-thinking_MI/datasets/ffhq/thumbnails128x128',
        transform: Optional[Callable] = None,
    ):
        super(FFHQ, self).__init__(root, transform=transform)
        # List to hold all file paths
        self.filename = []
        # Traverse subdirectories
        for subdir in os.listdir(root):
            subdir_path = os.path.join(root, subdir)
            if os.path.isdir(subdir_path):
                # Add images in this subdir to the filename list
                self.filename.extend([os.path.join(subdir_path, file) for file in os.listdir(subdir_path)])
    
    def __len__(self):
        return len(self.filename)
    
    def __getitem__(self, index: int):
        file_path = self.filename[index]
        # Check and switch file extension if necessary
        if not os.path.exists(file_path):
            file_path = file_path.replace('.jpg', '.png')
        
        # Load image
        im = PIL.Image.open(file_path)
        
        # Apply transformation if any
        if self.transform:
            im = self.transform(im)
        
        return im, -1

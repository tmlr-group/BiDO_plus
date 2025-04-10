import os
import PIL

from typing import Any, Callable, List, Optional, Union, Tuple
from torchvision.datasets import VisionDataset


class FFHQ(VisionDataset):
    """ 
    Modified CelebA dataset to adapt for custom cropped images.
    """

    def __init__(
            self,
            root='data/ffhq',
            transform: Optional[Callable] = None,
    ):
        super(FFHQ, self).__init__(root, transform=transform)
        self.filename = os.listdir(root)[:20000]

    def __len__(self):
        return len(self.filename)
    
    
    def __getitem__(self, index):
        file_path = os.path.join(self.root, self.filename[index])

        if os.path.exists(file_path) == False:
            file_path = file_path.replace('.jpg', '.png')
        im = PIL.Image.open(file_path)

        if self.transform:
            return self.transform(im)
        else:
            return im
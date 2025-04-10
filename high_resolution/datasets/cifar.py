from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms


class CIFAR100(Dataset):
    def __init__(self, 
                 mode, 
                 root_cifar="/home/comp/23481501/code/Defend_MI/Plug-and-Play-Attacks/data/cifar", 
                 transform=None):
        # Load CIFAR-100 dataset
        self.cifar100 = datasets.CIFAR100(root=root_cifar, train=(mode=='train'), transform=None, download=True)
        
        # Filter for the first 10 classes
        indices = [i for i, label in enumerate(self.cifar100.targets) if label < 10]
        self.data_subset = Subset(self.cifar100, indices)
        self.targets = [self.cifar100.targets[i] for i in indices]

        self.transform = transform

    def __len__(self):
        return len(self.data_subset)

    def __getitem__(self, idx):
        # Get image and label from the filtered CIFAR-100 dataset
        img, label = self.data_subset[idx]
        
        # Apply the transform to the image
        if self.transform:
            img = self.transform(img)

        return img, label



if __name__ == '__main__':
    # Define the transform to be applied to the images
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize the images to match the size of CIFAR-10 images
        transforms.ToTensor()
    ])

    # Initialize the CombinedDataset
    dataset = CIFAR100(mode="train", root_cifar="/home/comp/23481501/code/Defend_MI/Plug-and-Play-Attacks/data/cifar", transform=transform)
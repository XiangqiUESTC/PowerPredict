import torch
from torchvision import transforms, datasets

DATASET_INFO = {
            'cifar10': {
                'num_classes': 10,
                'transform': transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465),
                                         (0.2023, 0.1994, 0.2010))
                ])
            },
            'flowers102': {
                'num_classes': 102,
                'transform': transforms.Compose([
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406],
                                         [0.229, 0.224, 0.225])
                ])
            }
        }

def load_dataset(dataset_name, batch_size):
    ds = None
    """加载指定数据集"""
    if dataset_name == 'cifar10':
         ds = datasets.CIFAR10(
            root='../data/cifar10',
            train=False,
            download=True,
            transform=DATASET_INFO[dataset_name]['transform']
        )
    elif dataset_name == 'flowers102':
        ds =  datasets.Flowers102(
            root='../data',
            split='test',
            download=True,
            transform=DATASET_INFO[dataset_name]['transform']
        )
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader
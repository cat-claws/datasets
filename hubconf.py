# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch
from datasets import load_dataset

def loader(constructor, pretrained, **kwargs):
	model = constructor(**kwargs)
	if pretrained:
		checkpoint = f'https://github.com/cat-claws/nn/releases/download/parameters/{pretrained}.tar.gz'
		model.load_state_dict(torch.hub.load_state_dict_from_url(checkpoint, progress=True))
	return model

from torch.utils.data import TensorDataset

class TransformTensorDataset(TensorDataset):
    def __init__(self, *tensors, transform=None):
        super().__init__(*tensors)
        self.transform = transform

    def __getitem__(self, index):
        x, y = super().__getitem__(index)
        if self.transform:
            x = self.transform(x)
        return x, y
	    
def image_classification(transform, **kwargs):
	dataset = load_dataset(path, name, split)
	dataset.set_format(type="torch", columns=["image", "labels"])


def cifar(transform = None, **kwargs):
	return image_classification(transform, **kwargs)

from simple import Net, Net_, MultiLayerPerceptron, SimpleCNN
from convduonet import ConvDuoNet

def mlp(pretrained=False, **kwargs):
	return loader(MultiLayerPerceptron, pretrained=pretrained, **kwargs)
	

# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch
from datasets import load_dataset

# torch.hub.load('cat-claws/datasets', 'cifar10', name = 'cifar10-8', split='train', transform=transform)

# def loader(columns, **kwargs):
# 	dataset = load_dataset(**kwargs)
# 	dataset.set_format(type="numpy", columns=columns)
# 	return dataset

# Define retrieval transformation to output tuple
# def to_tuple(example):
#     image = example["image"]
#     label = example["labels"]
#     image = transform(image)
#     return (image, label)  # <- return tuple, not dict!

# def to_tuple(example):
#     image = example.values()
#     image = transform(image)
#     return (image, label)  # <- return tuple, not dict!
	
# # Apply it
# dataset = dataset.with_transform(to_tuple)

# def cifar10(transform=None, **kwargs):
# 	# def to_tuple(example):
# 	#         return (transform(example["image"]) if transform else example["image"], example['label'])
# 	dataset = load_dataset(**kwargs)
# 	dataset.set_format(type="numpy", columns= ["image", "label"])
# 	return dataset.with_transform(to_tuple)
	
# # 	return TransformTensorDataset(*tensors, transform=transform)

class CIFAR10(torch.utils.data.Dataset):
	def __init__(self, transform=None, **kwargs):
		self.hf_dataset = load_dataset(**kwargs)
		self.transform = transform

	def __getitem__(self, index):
		example = self.hf_dataset[index]
		return (self.transform(example["image"]) if self.transform else example["image"], example['label'])
		
	def __len__(self):
		return len(self.hf_dataset)
		
# class HFDataset(Dataset):
# 	def __init__(self, hf_dataset, transform=None, target_transform=None):
#         self.dataset = hf_dataset
#         self.transform = transform
#         self.target_transform = target_transform


# from torch.utils.data import TensorDataset

# class TransformTensorDataset(TensorDataset):
#     def __init__(self, *tensors, transform=None):
#         super().__init__(*tensors)
#         self.transform = transform

#     def __getitem__(self, index):
#         x, y = super().__getitem__(index)
#         if self.transform:
#             x = self.transform(x)
#         return x, y
	    
# def image_classification(transform, **kwargs):
# 	dataset = load_dataset(path, name, split)
# 	dataset.set_format(type="torch", columns=["image", "labels"])


# def cifar(transform = None, **kwargs):
# 	return image_classification(transform, **kwargs)

# from simple import Net, Net_, MultiLayerPerceptron, SimpleCNN
# from convduonet import ConvDuoNet

# def mlp(pretrained=False, **kwargs):
# 	return loader(MultiLayerPerceptron, pretrained=pretrained, **kwargs)
	

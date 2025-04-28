# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch
from datasets import load_dataset

class CIFAR10(torch.utils.data.Dataset):
	def __init__(self, transform=None, **kwargs):
		self.hf_dataset = load_dataset(**kwargs)
		self.transform = transform

	def __getitem__(self, index):
		example = self.hf_dataset[index]
		return (self.transform(example["image"]) if self.transform else example["image"], example['label'])
		
	def __len__(self):
		return len(self.hf_dataset)

# Optional list of dependencies required by the package
dependencies = ["torch"]

import torch
from datasets import load_dataset

class CIFAR10(torch.utils.data.Dataset):
	def __init__(self, transform=None, **kwargs):
		self.indexed = bool(kwargs.pop("indexed", False))
		self.hf_dataset = load_dataset(**kwargs)
		self.transform = transform

	def __getitem__(self, index):
		example = self.hf_dataset[index]
		x = example["image"]
		if self.transform:
			x = self.transform(x)
		if self.indexed:
			return (x, example['label'], index)
		else:
			return (x, example['label'])
	
	def __len__(self):
		return len(self.hf_dataset)

class TinyImagenet(CIFAR10):
	def __getitem__(self, index):
		example = self.hf_dataset[index]
		x = example["image"]
		if x.mode != "RGB":
			x = x.convert("RGB")
		if self.transform:
			x = self.transform(x)
		if self.indexed:
			return (x, example['label'], index)
		else:
			return (x, example['label'])

# datasets

```python
import torch
import torchvision

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_set = torch.hub.load('cat-claws/datasets', 'CIFAR10', path = 'cat-claws/poison', name = 'cifar10-16-huang2021unlearnable', split='train', transform = transform)

train_loader =  torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True, num_workers=8, pin_memory=True)

next(iter(train_loader))
```

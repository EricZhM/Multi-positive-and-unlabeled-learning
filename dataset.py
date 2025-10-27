import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np

class simple_dataset(Dataset):
    def __init__(self, data, labels):
        assert len(data) == len(labels)
        self.data = data
        self.targets = labels.long() if torch.is_tensor(labels) else torch.as_tensor(labels).long()

    def __getitem__(self, index):
        return self.data[index], int(self.targets[index])

    def __len__(self):
        return len(self.targets)


@torch.no_grad()
def as_tensor_batch(ds):
    if hasattr(ds, "data") and hasattr(ds, "targets"):
        data = ds.data
        targets = ds.targets
    elif hasattr(ds, "data") and hasattr(ds, "labels"):
        data = ds.data
        targets = ds.labels
    elif hasattr(ds, "tensors"):
        data, targets = ds.tensors
    else:
        xs, ys = [], []
        for i in range(len(ds)):
            x, y = ds[i]
            xs.append(torch.as_tensor(x))
            ys.append(int(y))
        data = torch.stack(xs, dim=0)
        targets = torch.as_tensor(ys)

    data = torch.as_tensor(data)
    targets = torch.as_tensor(targets)

    if not torch.is_floating_point(data):
        data = data.float()
    targets = targets.long()

    if data.ndim == 4 and data.shape[-1] in (1, 3):
        data = data.permute(0, 3, 1, 2).contiguous()

    return data, targets


class _MulticlassPUBuilder:
    def __init__(self, dataset_number: int, kprior: float, keep_special_case: bool = True, seed: int | None = None):
        assert dataset_number >= 2, "dataset_number 必须 >= 2"
        assert kprior > 0, "kprior 必须 > 0"
        self.K = int(dataset_number)
        self.U_label = self.K - 1
        self.kprior = float(kprior)
        self.keep_special = bool(keep_special_case)
        self.g = torch.Generator()
        if seed is not None:
            self.g.manual_seed(int(seed))

    @torch.no_grad()
    def build(self, dataset: Dataset):
        data, targets = as_tensor_batch(dataset)

        keep_mask = (targets >= 0) & (targets < self.K)
        data, targets = data[keep_mask], targets[keep_mask]

        if len(data) > 0:
            perm = torch.randperm(len(data), generator=self.g)
            data, targets = data[perm], targets[perm]

        idx_per_class = [torch.where(targets == c)[0] for c in range(self.K)]
        lenU_base = len(idx_per_class[self.U_label])

        kd24 = None
        if self.keep_special and (self.kprior == 0.2 and self.K == 4):
            lenU_base = int(0.2 * lenU_base)
            kd24 = lenU_base

        lenU = int(float(lenU_base) * (1.0 / self.kprior - 1.0) * (1.0 / float(self.K - 1)))

        pos_keep_chunks, pos_keep_labels = [], []
        U_chunks = []

        pri = torch.ones(self.K, dtype=torch.float64)

        for c in range(self.K - 1):
            idx = idx_per_class[c]
            if len(idx) == 0:
                pri[c] = 0.0
                continue
            take = min(lenU, len(idx))
            U_idx = idx[:take]
            keep_idx = idx[take:]

            pri[c] = float(len(U_idx))

            if len(keep_idx) > 0:
                pos_keep_chunks.append(data[keep_idx])
                pos_keep_labels.append(torch.full((len(keep_idx),), c, dtype=torch.long))
            if len(U_idx) > 0:
                U_chunks.append(data[U_idx])

        base_idx = idx_per_class[self.U_label]
        if kd24 is not None:
            base_idx = base_idx[:kd24]
        if len(base_idx) > 0:
            U_chunks.append(data[base_idx])
        pri[self.U_label] = float(len(base_idx))

        if len(pos_keep_chunks) > 0:
            pos_keep_data = torch.cat(pos_keep_chunks, dim=0)
            pos_keep_targets = torch.cat(pos_keep_labels, dim=0)
        else:
            pos_keep_data = data[:0]
            pos_keep_targets = torch.zeros((0,), dtype=torch.long)

        U_data = torch.cat(U_chunks, dim=0) if len(U_chunks) > 0 else data[:0]
        U_targets = torch.full((len(U_data),), self.U_label, dtype=torch.long)

        all_data = torch.cat([pos_keep_data, U_data], dim=0)
        all_targets = torch.cat([pos_keep_targets, U_targets], dim=0)
        if len(all_data) > 0:
            perm2 = torch.randperm(len(all_data), generator=self.g)
            all_data, all_targets = all_data[perm2], all_targets[perm2]

        if pri.sum() > 0:
            pri = torch.nn.functional.normalize(pri, p=1, dim=0)
        else:
            pri = torch.ones(self.K, dtype=torch.float64) / float(self.K)

        return simple_dataset(all_data, all_targets), pri


class CustomDataset(Dataset):
    def __init__(self, dataset_name, train_flag, dataset_number, kprior, root, seed: int | None = 42):
        self.dataset_name = dataset_name
        self.dataset_number = int(dataset_number)
        self.kprior = float(kprior)
        self.pri = None
        self.root = root

        train_base, test_base = self._load_base(dataset_name, train_flag)

        train_dataset = self._pack_first_K_classes(train_base, K=self.dataset_number)
        test_dataset = self._pack_first_K_classes(test_base, K=self.dataset_number)

        builder = _MulticlassPUBuilder(dataset_number=self.dataset_number,
                                       kprior=self.kprior,
                                       keep_special_case=True,
                                       seed=seed)
        train_dataset, self.pri = builder.build(train_dataset)

        self.traindataset = train_dataset
        self.testdataset = test_dataset

    def __call__(self):
        return self.traindataset, self.testdataset

    def get_pri(self):
        print(self.pri)
        return self.pri

    def _load_base(self, name: str, train_flag: bool):
        if name == 'MNIST':
            self.is_uci = False
            train = datasets.MNIST(root=self.root, train=True, download=True, transform=ToTensor())
            test  = datasets.MNIST(root=self.root, train=False, download=True, transform=ToTensor())
        elif name == 'FashionMNIST':
            self.is_uci = False
            train = datasets.FashionMNIST(root=self.root, train=True, download=True, transform=ToTensor())
            test  = datasets.FashionMNIST(root=self.root, train=False, download=True, transform=ToTensor())
        elif name == 'USPS':
            self.is_uci = False
            train = datasets.USPS(root=self.root, train=True, download=True, transform=ToTensor())
            test  = datasets.USPS(root=self.root, train=False, download=True, transform=ToTensor())
        elif name == 'KMNIST':
            self.is_uci = False
            train = datasets.KMNIST(root=self.root, train=True, download=True, transform=ToTensor())
            test  = datasets.KMNIST(root=self.root, train=False, download=True, transform=ToTensor())
        elif name == 'SVHN':
            self.is_uci = False
            train = datasets.SVHN(root=self.root, split='train', download=True, transform=ToTensor())
            test  = datasets.SVHN(root=self.root, split='test',  download=True, transform=ToTensor())
        else:
            raise NotImplementedError(f"Unsupported dataset: {name}")
        return train, test

    @torch.no_grad()
    def _pack_first_K_classes(self, base_ds, K: int):
        data, targets = as_tensor_batch(base_ds)
        mask = (targets >= 0) & (targets < K)
        data = data[mask]; targets = targets[mask]
        return simple_dataset(data, targets)

    
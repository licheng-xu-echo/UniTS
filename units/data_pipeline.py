from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import logging
import numpy as np
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import DataLoader

from .data import MultiDataset, MultiDataset1x, MultiDatasetV2, get_fnidx_split, get_idx_split
from .registry import Registry

DATASET_BUILDERS: Registry = Registry("dataset_builder")


@DATASET_BUILDERS.register_fn("dataset_v1")
def _build_dataset_v1(args, name_regrex: str):
    return MultiDataset(root=args.dataset_path, name_regrex=name_regrex)


@DATASET_BUILDERS.register_fn("dataset_v2")
def _build_dataset_v2(args, name_regrex: str):
    return MultiDatasetV2(
        root=args.dataset_path,
        name_regrex=name_regrex,
        link_rc=getattr(args, "link_rc", False),
        data_enhance=getattr(args, "data_enhance", False),
    )


@DATASET_BUILDERS.register_fn("dataset_1x")
def _build_dataset_1x(args, name_regrex: str):
    return MultiDataset1x(
        root=args.dataset_path,
        name_regrex=name_regrex,
        link_rc=getattr(args, "link_rc", False),
        iptgraph_type=getattr(args, "iptgraph_type", "rct+pdt"),
    )


_TYPE_TO_KEY = {
    1: "dataset_v1",
    2: "dataset_v2",
    3: "dataset_1x",
}


@dataclass
class DataLoaders:
    train: DataLoader
    valid: DataLoader
    test: DataLoader
    train_sampler: Optional[DistributedSampler] = None
    valid_sampler: Optional[DistributedSampler] = None
    test_sampler: Optional[DistributedSampler] = None


def register_dataset_builder(name: str, builder) -> None:
    DATASET_BUILDERS.register(name, builder)


def resolve_dataset_builder_key(args) -> str:
    if hasattr(args, "dataset_builder") and getattr(args, "dataset_builder"):
        return str(getattr(args, "dataset_builder"))
    dataset_type = int(getattr(args, "dataset_type", 1))
    if dataset_type not in _TYPE_TO_KEY:
        raise ValueError(f"Unsupported dataset_type={dataset_type}")
    return _TYPE_TO_KEY[dataset_type]


def build_dataset_from_args(args, name_regrex: Optional[str] = None):
    key = resolve_dataset_builder_key(args)
    builder = DATASET_BUILDERS.get(key)
    pattern = name_regrex or getattr(args, "name_regrex", "dataset_0_*.npy")
    return builder(args, pattern)


def shuffle_and_truncate_dataset(dataset, args):
    sf_index = list(range(len(dataset)))
    np.random.seed(getattr(args, "seed", 2024))
    np.random.shuffle(sf_index)

    data_truncated = int(getattr(args, "data_truncated", 0))
    if data_truncated > 0:
        return dataset[sf_index[:data_truncated]]
    return dataset[sf_index]


def _split_dataset_indices(dataset, args):
    train_ratio = float(getattr(args, "train_ratio", 0.8))
    valid_ratio = float(getattr(args, "valid_ratio", 0.1))
    seed = int(getattr(args, "seed", 2024))
    data_enhance = bool(getattr(args, "data_enhance", False))

    if not data_enhance:
        return get_idx_split(
            len(dataset),
            int(train_ratio * len(dataset)),
            int(valid_ratio * len(dataset)),
            seed,
        )

    # materialize first item to ensure aggregated attributes are ready
    dataset[0]
    return get_fnidx_split(dataset.data.fn_id, train_ratio, valid_ratio, seed)


def build_train_valid_test_datasets(args, rank: int = 0):
    dataset = build_dataset_from_args(args, getattr(args, "name_regrex", None))
    dataset = shuffle_and_truncate_dataset(dataset, args)

    specific_test = bool(getattr(args, "specific_test", False))

    if not specific_test:
        if rank == 0:
            logging.info("[INFO] Splitting dataset into train/valid/test")
            logging.info("[INFO] Dataset size: %d", len(dataset))
        split_ids_map = _split_dataset_indices(dataset, args)
        return (
            dataset[split_ids_map["train"]],
            dataset[split_ids_map["valid"]],
            dataset[split_ids_map["test"]],
        )

    test_name_regrex = getattr(args, "test_name_regrex", "")
    if not test_name_regrex:
        raise ValueError("specific_test=True requires non-empty test_name_regrex")
    train_ratio = float(getattr(args, "train_ratio", 0.8))
    valid_ratio = float(getattr(args, "valid_ratio", 0.1))
    if abs(train_ratio + valid_ratio - 1.0) > 1e-6:
        raise ValueError("specific_test=True requires train_ratio + valid_ratio == 1.0")

    if rank == 0:
        logging.info("[INFO] Splitting dataset into train/valid with specified test set")

    split_ids_map = _split_dataset_indices(dataset, args)
    train_dataset = dataset[split_ids_map["train"]]
    valid_dataset = dataset[split_ids_map["valid"]]
    test_dataset = build_dataset_from_args(args, test_name_regrex)
    return train_dataset, valid_dataset, test_dataset


def build_test_dataset_for_sampling(args):
    dataset = build_dataset_from_args(args, getattr(args, "name_regrex", None))
    dataset = shuffle_and_truncate_dataset(dataset, args)

    specific_test = bool(getattr(args, "specific_test", False))
    if not specific_test:
        split_ids_map = _split_dataset_indices(dataset, args)
        return dataset[split_ids_map["test"]]

    test_name_regrex = getattr(args, "test_name_regrex", "")
    if not test_name_regrex:
        raise ValueError("specific_test=True requires non-empty test_name_regrex")
    train_ratio = float(getattr(args, "train_ratio", 0.8))
    valid_ratio = float(getattr(args, "valid_ratio", 0.1))
    if abs(train_ratio + valid_ratio - 1.0) > 1e-6:
        raise ValueError("specific_test=True requires train_ratio + valid_ratio == 1.0")
    return build_dataset_from_args(args, test_name_regrex)


def build_dataloaders(
    args,
    train_dataset,
    valid_dataset,
    test_dataset,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    batch_size = int(getattr(args, "batch_size", 32))
    num_workers = int(getattr(args, "num_workers", 0))
    test_reduce_ratio = int(getattr(args, "test_reduce_ratio", 1))
    test_batch_size = max(1, batch_size // max(1, test_reduce_ratio))

    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
        valid_sampler = DistributedSampler(
            valid_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )
        test_sampler = DistributedSampler(
            test_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
        )

        return DataLoaders(
            train=DataLoader(
                train_dataset,
                batch_size=batch_size,
                sampler=train_sampler,
                num_workers=num_workers,
                pin_memory=True,
            ),
            valid=DataLoader(
                valid_dataset,
                batch_size=batch_size,
                sampler=valid_sampler,
                num_workers=num_workers,
                pin_memory=True,
            ),
            test=DataLoader(
                test_dataset,
                batch_size=test_batch_size,
                sampler=test_sampler,
                num_workers=num_workers,
                pin_memory=True,
            ),
            train_sampler=train_sampler,
            valid_sampler=valid_sampler,
            test_sampler=test_sampler,
        )

    return DataLoaders(
        train=DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        valid=DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
        test=DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=num_workers),
    )

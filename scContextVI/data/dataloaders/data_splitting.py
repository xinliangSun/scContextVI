import numpy as np

from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter
from scvi.dataloaders._data_splitting import (
    validate_data_split,
    validate_data_split_with_external_indexing,
)

from scContextVI.data.dataloaders.scContextVI_dataloader import scContextDataLoader


class scContextVIDataSplitter(DataSplitter):
    """
    Create scContextVIDataLoader for training, validation, and test set.

    Args:
    ----
        adata_manager: `~scvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
        group_indices_list: List where each element is a list of indices in the adata to load.
        train_size: Proportion of data to include in the training set.
        validation_size: Proportion of data to include in the validation set. The
            remaining proportion after `train_size` and `validation_size` is used for
            the test set.
        accelerator: Use default CPU or GPU if available.
        **kwargs: Keyword args for data loader (`ContrastiveDataLoader`).
    """

    def __init__(
            self,
            adata_manager: AnnDataManager,
            group_indices_list: list[list[int]],
            train_size: float | None = None,
            validation_size: float | None = None,
            batch_size: int = 128,
            shuffle_set_split: bool = True,
            load_sparse_tensor: bool = False,
            pin_memory: bool = False,
            external_indexing: list[np.array, np.array, np.array] | None = None,
            **kwargs,
    ) -> None:
        super().__init__(
            adata_manager=adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            shuffle_set_split=shuffle_set_split,
            load_sparse_tensor=load_sparse_tensor,
            pin_memory=pin_memory,
            **kwargs
        )
        self.train_idx_per_group = None
        self.val_idx_per_group = None
        self.test_idx_per_group = None
        self.adata_manager = adata_manager
        self.group_indices_list = group_indices_list
        self.train_size = train_size
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.data_loader_kwargs = kwargs

        self.n_per_group = [len(group_indices) for group_indices in self.group_indices_list]
        n_train_per_group = []
        n_val_per_group = []

        for group_indices in self.group_indices_list:
            n_train, n_val = validate_data_split(
                len(group_indices), self.train_size, self.validation_size
            )
            n_train_per_group.append(n_train)
            n_val_per_group.append(n_val)

        self.n_val_per_group = n_val_per_group
        self.n_train_per_group = n_train_per_group

        self.current_dataloader = None

    def setup(self, stage: str | None = None):
        random_state = np.random.RandomState(seed=settings.seed)

        self.train_idx_per_group = []
        self.val_idx_per_group = []
        self.test_idx_per_group = []

        for i, group_indices in enumerate(self.group_indices_list):
            group_permutation = random_state.permutation(group_indices)
            n_train_group = self.n_train_per_group[i]
            n_val_group = self.n_val_per_group[i]

            self.val_idx_per_group.append(group_permutation[:n_val_group])
            self.train_idx_per_group.append(
                group_permutation[n_val_group: (n_val_group + n_train_group)]
            )
            self.test_idx_per_group.append(
                group_permutation[(n_train_group + n_val_group):]
            )

        self.train_idx = np.concatenate(self.train_idx_per_group)
        self.val_idx = np.concatenate(self.val_idx_per_group)
        self.test_idx = np.concatenate(self.test_idx_per_group)

    def train_dataloader(self) -> scContextDataLoader:
        return scContextDataLoader(
            adata_manager=self.adata_manager,
            indices_list=self.train_idx_per_group,
            shuffle=True,
            batch_size=self.batch_size,
            drop_last=self.drop_last,  # self.drop_last
            load_sparse_tensor=self.load_sparse_tensor,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self) -> scContextDataLoader:
        if np.all([len(val_idx) > 0 for val_idx in self.val_idx_per_group]):
            return scContextDataLoader(
                adata_manager=self.adata_manager,
                indices_list=self.val_idx_per_group,
                shuffle=False,
                batch_size=self.batch_size,
                drop_last=False,
                load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self) -> scContextDataLoader:
        if np.all([len(test_idx) > 0 for test_idx in self.test_idx_per_group]):
            return scContextDataLoader(
                adata_manager=self.adata_manager,
                indices_list=self.test_idx_per_group,
                shuffle=False,
                batch_size=self.batch_size,
                drop_last=False,
                load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
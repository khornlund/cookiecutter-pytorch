from torch.utils.data import DataLoader


class DataLoaderBase(DataLoader):
    """
    Base class for all data loaders
    """

    def split_validation(self) -> DataLoader:
        """
        Return a `torch.utils.data.DataLoader` for validation, or None if not available.
        """
        raise NotImplementedError

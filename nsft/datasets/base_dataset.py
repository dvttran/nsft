from pathlib import Path
from typing import Union, Callable


class BaseDataset:
    _datasets = dict()

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseDataset._datasets[name.lower()] = cls
            cls._name = name
        else:
            BaseDataset._datasets[cls.__name__.lower()] = cls
            cls._name = cls.__name__

    def __init__(self,
        root_dir: Union[Path, str] = None,
        dataset_name: str = '',
        sequence_type: str = '',
        sequence_name: str = '',
        generator: Callable = None,
        **kwargs
    ):
        assert root_dir is not None or generator is not None, "Either root_dir or generator must be provided"

        if root_dir:
            if isinstance(root_dir, str):
                root_dir = Path(root_dir)
            data_dir = Path(root_dir).joinpath(dataset_name, sequence_type, sequence_name).as_posix()
        else:
            data_dir = None

        self.dataset_name = dataset_name
        self.sequence_type = sequence_type
        self.sequence_name = sequence_name
        self.root_dir = root_dir
        self.data_dir = data_dir
        self.generator = generator
        self.dataset = None
        self.render_fn = None

    def _read_data(self, *args, **kwargs):
        pass

    def _generate_data(self, *args, **kwargs):
        pass

    def read_data(self, *args, **kwargs):
        if self.data_dir:
            return self._read_data(*args, **kwargs)
        else:
            return self._generate_data(*args, **kwargs)

    def save_data(self, *args, **kwargs):
        pass


def available_datasets():
    return list(BaseDataset._datasets.keys())


def get_dataset(dataset_config: dict, device="cpu"):
    name = dataset_config.pop("name").lower()
    try:
        return BaseDataset._datasets[name](device=device, **dataset_config)
    except KeyError:
        raise ValueError(f"Got dataset {name}, expected one of {available_datasets()}.")

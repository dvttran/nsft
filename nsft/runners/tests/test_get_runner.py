import pytest
from ..base_runner import get_runner
from pathlib import Path


@pytest.mark.parametrize("config_file", ["Phi_SfT.yaml"])
def test_get_runner(config_file):
    cwd = Path("/Users/thuytdv/github/nsft")
    config_folder = cwd.joinpath("configs/runners")
    runner = get_runner(config_file=config_folder.joinpath(config_file))
    # runner.run() # might be slow

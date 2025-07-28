import pytest
import os
from ..base_dataset import get_dataset
from ..sft import SfTDataset
import yaml
import matplotlib.pyplot as plt


@pytest.mark.parametrize("config_file", ["kinect_paper.yaml", "sft_real.yaml", "sft_synthetic.yaml"])
def test_get_dataset(config_file):
    cwd = "/Users/thuytdv/github/nsft"
    # Load model and implicitly reshape x0
    config_path = os.path.join(cwd, "configs/datasets", config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset_config = config["dataset"]
    dataset_config["root_dir"] = os.path.join(cwd, dataset_config["root_dir"])
    dataset = get_dataset(dataset_config)
    if isinstance(dataset, SfTDataset):
        verts_padded = dataset.get_vertices(dataset.x0)

        render_rgb, render_sil = dataset.render_fn(verts_padded=verts_padded)
        render_rgb, render_sil, render_depth, render_mask = dataset.render_with_depth_fn(verts_padded=verts_padded)

        assert render_rgb.ndim == 4 and render_sil.ndim == 4 and render_depth.ndim == 4 and render_mask == 4


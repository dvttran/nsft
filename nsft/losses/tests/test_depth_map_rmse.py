import pytest
import os

import torch

from ...datasets.base_dataset import get_dataset
from ...datasets.sft import SfTDataset
from ...losses import Losses
import yaml
from loguru import logger

@pytest.mark.parametrize("config_file", ["kinect_paper.yaml"])
def test_depth_map_rmse(config_file):
    # Depth Map RMSE Metric
    metric_name = "depth_rmse"
    metric_config = {
        "name": metric_name,
        "hyperparams": {
            "scale": 1.0,
            "inpaint_radius": 3,
            "with_boundary": False,
        }
    }
    depth_map_rmse_metric = Losses(metric_config)
    args = {}

    cwd = "/Users/thuytdv/github/nsft"
    # Load model and implicitly reshape x0
    config_path = os.path.join(cwd, "configs/datasets", config_file)
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    dataset_config = config["dataset"]
    dataset_config["root_dir"] = os.path.join(cwd, dataset_config["root_dir"])
    dataset = get_dataset(dataset_config)
    if isinstance(dataset, SfTDataset):
        # verts_padded = dataset.get_vertices(dataset.x0)

        meshes = dataset._read_meshes_from_dir("/Users/thuytdv/Downloads/surfaces-2")
        verts_padded = (dataset.inverse_align_to_gt(meshes.verts_padded()[28:32]))

        render_rgb, render_sil, render_depth, render_mask = dataset.render_with_depth_fn(verts_padded=verts_padded)

        args.update({
            "render_depth": render_depth,
            "render_mask": render_mask,
            "gt_depth": dataset.gt_kinect_data[28:32, ..., 2:],
            "gt_mask": dataset.gt_mask[28:32],
        })
        errors = depth_map_rmse_metric(**args)
        sequence_error = errors.mean()
        logger.info(
            f"\n{metric_name.capitalize()} Error:\n" +
            "\n".join([
                f" Frame {frame_idx}: {e.item()}" for frame_idx, e in enumerate(errors)
            ]) +
            f"\nSequence Error: {sequence_error}"
        )


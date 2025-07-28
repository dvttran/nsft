import sys
from pathlib import Path
from typing import Union
# import yaml
from configmypy import ConfigPipeline, YamlConfig, ArgparseConfig
import torch
import random
import numpy as np
from loguru import logger
from ..datasets import get_dataset
from ..models import get_model
from .optimizer import get_optimizer
from ..losses import Losses
# Import wandb if usage
wandb_available = False
try:
    import wandb
    wandb_available = True
except ModuleNotFoundError:
    wandb_available = False


class BaseRunner:
    _runners = dict()

    def __init_subclass__(cls, name=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if name is not None:
            BaseRunner._runners[name.lower()] = cls
            cls._name = name
        else:
            BaseRunner._runners[cls.__name__.lower()] = cls
            cls._name = cls.__name__

    def __init__(
        self,
        log_every: int = 1,
        seed: int = 42,
        device: Union[torch.device, str] = "cpu",
        wandb_config: dict = None,
        dataset_config: dict = None,
        model_config: dict = None,
        optimizer_config: dict = None,
        losses_config: dict = None,
        metrics_config: dict = None,
        **kwargs
    ):
        # Logging
        self.log_every = log_every

        # Set seed
        self.seed = seed
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        # device
        self.device = device

        # wandb
        if wandb_available and wandb_config is not None:
            wandb_config.update(kwargs)
            self.wandb_init(**wandb_config, tags=["runner"])
            self.wandb_available = True
        else:
            self.wandb_available = False

        # dataset
        self.dataset = get_dataset(dataset_config, device=device)
        self.x0 = self.dataset.x0
        self.time_span = self.dataset.time_span
        self.data_channels = self.dataset.data_channels
        self.get_vertices = lambda x: self.dataset.get_vertices(x)
        self.align_to_gt = lambda verts_padded: self.dataset.align_to_gt(verts_padded)

        # render
        if self.dataset.render_fn is not None:
            self.render_fn = self.dataset.render_fn
        if self.dataset.render_with_depth_fn is not None:
            self.render_with_depth_fn = self.dataset.render_with_depth_fn

        # model
        self.model = get_model(model_config["config_path"], self.data_channels)
        self.model.to(device)

        # optimizer
        self.get_optimizer = lambda params: get_optimizer(params, **optimizer_config)
        self.optimizer = get_optimizer(self.model.parameters(), **optimizer_config)

        # losses
        self.losses_name = [loss_config["name"] for loss_config in losses_config]
        kwargs = {
            "edge_lengths_0": self.dataset.edge_lengths_0,  # adaptive mesh_inextensibility loss
            "device": device,  # perceptual loss
        }
        self.losses = Losses(losses_config, **kwargs)

        # metrics
        if metrics_config is not None:
            self.metrics = dict()
            self.metrics_name = []
            for metric_config in metrics_config:
                metric_name = metric_config["name"].lower()
                self.metrics.update({metric_name: Losses(metric_config)})
                self.metrics_name.append(metric_name)

    def wandb_init(self, dir="./", name="runner", project="SfT", mode="offline", tags=None, **kwargs):
        wandb.init(
            dir=dir,
            project=project,
            name=name,
            mode=mode,
            config=kwargs,
            tags=tags
        )
        logger.info(f"wandb dir: {wandb.run.dir}")

    def check_loss(self, loss_name):
        return any([name == loss_name for name in self.losses_name])

    def check_metric(self, metric_name):
        return any([name == metric_name for name in self.metrics_name])

    def evaluate(self, x_pred, log=True, **kwargs):
        args = dict()
        if self.check_metric("chamfer"):
            metric_name = "chamfer"
            args.update({
                "verts_padded": self.align_to_gt(self.get_vertices(x_pred)),
                "faces_packed": self.dataset.template.faces_packed(),
                "gt_point_clouds": self.dataset.gt_point_clouds,
                "gt_point_clouds_lengths": self.dataset.gt_point_clouds_lengths,
                "min_index": 0,
                "max_index": self.dataset.n_time_span,
            })
        elif self.check_metric("lp"):
            metric_name = "lp"
            args.update({
                "x_pred": x_pred.squeeze(),
                "x": self.dataset.x.squeeze(),
            })
        elif self.check_metric("depth_rmse"):
            metric_name = "depth_rmse"
            render_rgb, render_sil, render_depth, render_mask = self.render_with_depth_fn(self.get_vertices(x_pred))
            args.update({
                "render_depth": render_depth,
                "render_mask": render_mask,
                "gt_depth": self.dataset.gt_kinect_data[..., 2:],
                "gt_mask": self.dataset.gt_mask,
            })
        else:
            raise ValueError(f"Please specify metrics in the config file.")

        try:
            metric = self.metrics[metric_name]
        except KeyError:
            raise KeyError(f"Please specify hyper-parameters for metric `{metric_name}` in the config file.")
        errors = metric(**args)
        sequence_error = errors.mean()
        if log:
            logger.info(
                f"\n{metric_name.capitalize()} Error:\n" +
                "\n".join([
                    f" Frame {frame_idx}: {e.item()}" for frame_idx, e in enumerate(errors)
                ]) +
                f"\nSequence Error: {sequence_error}"
            )
        return sequence_error

    def evaluate_surface_isometry(self, output_path, **kwargs):
        meshes = self.dataset.read_output_meshes(output_path=output_path, **kwargs)
        verts_padded = meshes.verts_padded()
        edges_packed = self.dataset.edges_packed
        faces_packed = meshes[0].faces_packed()

        # compute edge length
        edges_lengths_0 = self.dataset.edge_lengths_0
        edges_lengths = self.dataset.mesh.get_edge_lengths(verts_padded=verts_padded, edges_packed=edges_packed)

        # compute edge differences
        edges_diffs = torch.abs(edges_lengths - edges_lengths_0) / edges_lengths_0 # relative error
        repeated_edges_diffs = torch.repeat_interleave(edges_diffs, 2, dim=1)

        # gather edge differences for vertices
        verts_ids = edges_packed.view(-1).expand(len(verts_padded), -1).to(torch.int64)
        verts_edge_diffs = torch.zeros_like(verts_padded[..., 0])
        verts_edge_diffs.scatter_add_(dim=1, index=verts_ids, src=repeated_edges_diffs)

        # save eval meshes
        self.dataset.save_eval_meshes(
            verts_padded=verts_padded,
            faces_packed=faces_packed,
            eval_scalar_field_padded=verts_edge_diffs,
            output_path=output_path,
            **kwargs
        )

    def evaluate_render_image(self, output_path, **kwargs):
        if isinstance(output_path, str):
            output_path = Path(output_path)
        output_path = output_path.joinpath(
            self.dataset.dataset_name,
            self.dataset.sequence_type,
            self.dataset.sequence_name,
        )
        images_dir = output_path.joinpath("rgbs")
        render_rgb = self.dataset._read_video_from_dir(images_dir=images_dir) / 255.
        target_rgb = self.dataset.target_rgb.cpu().numpy()

        rgb_diff = np.abs(render_rgb - target_rgb)

        # save rgb diff
        images_diff_dir = output_path.joinpath("rgbs_diff")
        self.dataset.save_images(
            images_dir=images_diff_dir,
            images=rgb_diff,
            **kwargs
        )
        # images_diff_dir = output_path.joinpath("rgbs_diff")
        # Path.mkdir(images_diff_dir, parents=True, exist_ok=True)
        # for idx, rgb in enumerate(render_rgb):
        #     self._save_image(os.path.join(images_diff_dir, f"{rgbs_name + '_' if rgbs_name else ''}{idx:03d}.png"), rgb)



    def save_outputs(self, x_pred, output_path, **kwargs):
        verts_padded = self.get_vertices(x_pred)
        return self.dataset.save_data(verts_padded=verts_padded, output_path=output_path, **kwargs)

    def run(self, **kwargs):
        pass

    def window_step(self, x_window, n_windows_current, warmup: bool, window_size=3):
        # Loss
        args = dict()

        # Visual loss
        if self.check_loss("rgb") or self.check_loss("sil") or self.check_loss("image_gradient") or self.check_loss("perceptual"):
            render_rgb, render_sil = self.render_fn(self.get_vertices(x_window))
            target_rgb = self.dataset.target_rgb[n_windows_current - 1:n_windows_current - 1 + window_size]
            target_sil = self.dataset.target_sil[n_windows_current - 1:n_windows_current - 1 + window_size]
            args.update({
                "render_rgb": render_rgb,
                "render_sil": render_sil,
                "target_rgb": target_rgb,
                "target_sil": target_sil,
            })

        # mesh_inextensibility loss
        if self.check_loss("mesh_inextensibility") or self.check_loss("adaptive_mesh_inextensibility"):
            args.update({
                "verts_padded": self.get_vertices(x_window),
                "eig_vals": self.dataset.eig_vals,
                "adj_mat": self.dataset.adj_mat,
            })

        # Edge loss
        if self.check_loss("edge"):
            args.update({
                "verts_padded": self.get_vertices(x_window),
                "edges_packed": self.dataset.edges_packed,
                "edge_lengths_0": self.dataset.edge_lengths_0,
            })

        # Thin shell loss
        if self.check_loss("thin_shell"):
            args.update({
                "verts_padded": self.get_vertices(x_window),
                "verts_packed_0": self.get_vertices(self.dataset.x0),
                "faces_packed": self.dataset.mesh.faces_packed,
                "E": self.dataset.mesh.E,
                "EMAP": self.dataset.mesh.EMAP,
                "EF": self.dataset.mesh.EF,
                "EI": self.dataset.mesh.EI,
            })

        # temporal loss: x1 = (x0 + x2) / 2
        x0, x1, x2 = x_window.unbind(dim=0)
        temporal_loss = (
            torch.norm(x1 - (x0 + x2) / 2, p=2, dim=-1) / torch.norm(x1, p=2, dim=-1)
        ).mean()

        # Total loss
        loss = self.losses(warmup=warmup, **args)

        return loss, temporal_loss

    def step(self, x1, n_frames_current, warmup: bool):
        # Loss
        args = dict()

        # Visual loss
        if self.check_loss("rgb") or self.check_loss("sil") or self.check_loss("image_gradient") or self.check_loss("perceptual"):
            render_rgb, render_sil = self.render_fn(self.get_vertices(x1))
            target_rgb = self.dataset.target_rgb[n_frames_current - 1:n_frames_current]
            target_sil = self.dataset.target_sil[n_frames_current - 1:n_frames_current]
            args.update({
                "render_rgb": render_rgb,
                "render_sil": render_sil,
                "target_rgb": target_rgb,
                "target_sil": target_sil,
            })

        # mesh_inextensibility loss
        if self.check_loss("mesh_inextensibility") or self.check_loss("adaptive_mesh_inextensibility"):
            args.update({
                "verts_padded": self.get_vertices(x1),
                "eig_vals": self.dataset.eig_vals,
                "adj_mat": self.dataset.adj_mat,
            })

        # Edge loss
        if self.check_loss("edge"):
            args.update({
                "verts_padded": self.get_vertices(x1),
                "edges_packed": self.dataset.edges_packed,
                "edge_lengths_0": self.dataset.edge_lengths_0,
            })

        # Thin shell loss
        if self.check_loss("thin_shell"):
            args.update({
                "verts_padded": self.get_vertices(x1),
                "verts_packed_0": self.get_vertices(self.dataset.x0),
                "faces_packed": self.dataset.mesh.faces_packed,
                "E": self.dataset.mesh.E,
                "EMAP": self.dataset.mesh.EMAP,
                "EF": self.dataset.mesh.EF,
                "EI": self.dataset.mesh.EI,
            })

        # Gravity loss
        if self.check_loss("gravity"):
            args.update({
                "verts_padded": self.get_vertices(x1),
            })

        # Total loss
        loss = self.losses(warmup=warmup, **args)

        return loss


def available_runners():
    return list(BaseRunner._runners.keys())

def get_runner(
        config_file: Union[str, Path],
        config_name: str = "default",
):
    # with open(config_path, "r") as f:
    #     config = yaml.safe_load(f)
    pipe = ConfigPipeline([
        YamlConfig(config_file=config_file, config_name=config_name),
        ArgparseConfig(config_file=None, config_name=None),  # NOTE: comment this line for pytest
        YamlConfig(),
    ])
    config = pipe.read_conf()
    pipe.log()
    logger.info(f"Config: {config}")
    runner_config = config["runner"]
    name = runner_config.pop("name").lower()

    # Parse config
    log_every = runner_config.pop("log_every", 1)
    seed = runner_config.pop("seed", 42)
    device = runner_config.pop("device", "cuda" if torch.cuda.is_available() else "cpu")
    wandb_config = runner_config.get("wandb", None)
    dataset_config = runner_config.get("dataset", None)
    model_config = runner_config.get("model", None)
    optimizer_config = runner_config.get("optimizer", None)
    losses_config = runner_config.get("losses", None)
    metrics_config = runner_config.get("metrics", None)

    try:
        runner = BaseRunner._runners[name](
            log_every=log_every,
            seed=seed,
            device=device,
            wandb_config=wandb_config,
            dataset_config=dataset_config,
            model_config=model_config,
            optimizer_config=optimizer_config,
            losses_config=losses_config,
            metrics_config=metrics_config,
            **runner_config
        )
        # TODO: check renderer.world_to_ndc_transform.get_matrix()
        return runner
    except KeyError:
        raise ValueError(f"Got runner name={name}, expected one of {available_runners()}.")



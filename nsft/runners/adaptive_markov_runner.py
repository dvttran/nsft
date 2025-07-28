from .base_runner import BaseRunner, wandb_available
import time
from loguru import logger
if wandb_available:
    import wandb
import torch
from typing import Union
from pathlib import Path


class AdaptiveMarkovRunner(BaseRunner, name="AdaptiveMarkov"):
    def __init__(self, n_epochs_warmup: int = 0, n_epochs_per_frame: int = 500, tolerance: float = 1e-6, **kwargs):
        super().__init__(**kwargs)
        self.n_epochs_warmup = n_epochs_warmup
        self.n_epochs_per_frame = n_epochs_per_frame
        self.n_frames = self.dataset.n_time_span
        self.n_epochs = n_epochs_warmup + n_epochs_per_frame * self.n_frames
        self.tolerance = tolerance
        logger.info(f"Adaptive runner tolerance: {self.tolerance}")

    def _step(self, x1, n_frames_current):
        # Loss
        args = dict()

        # Visual loss
        if self.check_loss("rgb") or self.check_loss("sil"):
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

        # Total loss
        loss = self.losses(**args)

        return loss

    def run(self,
        output_path: Union[str, Path] = "outputs/",
        surfaces_name: str = 'surface',
        rgbs_name: str = 'rgb',
        blurred_masks_name: str = None,
        materials: bool = False,
        **kwargs
    ):
        # Learning
        x0 = self.x0.clone()
        x_pred = self.x0.clone()[None]
        n_frames_current = 1

        opt_start = time.time()
        curr_loss = -1.0
        for epoch in range(self.n_epochs):
            # Meta data Warmup logging
            if epoch == 0 and self.n_epochs_warmup > 0:
                logger.info(f"<-- Warmup {self.n_epochs_warmup} epochs starting ... -->")
                logger.info(f"<-- Start frame {n_frames_current - 1} -->")
            if epoch == self.n_epochs_warmup:
                if self.n_epochs_warmup > 0:
                    logger.info(f"<-- Warmup end ! -->")
                logger.info(f"<-- Main run {self.n_epochs_per_frame} epochs/frame starting ... -->")

            start = time.time()
            n_iters = 0
            while True:
                self.optimizer.zero_grad()
                x1 = (x0 + self.model(x0))[None]
                loss = self._step(x1, n_frames_current)
                # Back propagation
                loss.backward()
                self.optimizer.step()
                n_iters += 1
                if abs(loss - curr_loss) < self.tolerance:
                    break
                curr_loss = loss.detach().clone().item()
            end = time.time()
            logger.info(f"Epoch {((epoch - self.n_epochs_warmup)  % (self.n_epochs_per_frame) + 1)}/{self.n_epochs_per_frame}: {str(self.losses)}, Loss: {loss.item():.4f} | Run {n_iters} iterations | Runtime: {end - start:.4f}s")


            # # Warm up logging
            # if epoch < self.n_epochs_warmup and (epoch == 0 or (epoch + 1) % self.log_every == 0 or epoch == self.n_epochs_warmup - 1):
            #     logger.info(f"Warmup {epoch + 1}/{self.n_epochs_warmup}: {str(self.losses)}, Loss: {loss.item():.4f} | Runtime: {end - start:.4f}s")
            #
            # # Main run logging
            # if epoch >= self.n_epochs_warmup and ((epoch - self.n_epochs_warmup) % self.n_epochs_per_frame == 0 or (epoch + 1) % self.log_every == 0 or (epoch - self.n_epochs_warmup) % self.n_epochs_per_frame == self.n_epochs_per_frame - 1):
            #     logger.info(f"Epoch {((epoch - self.n_epochs_warmup)  % (self.n_epochs_per_frame) + 1)}/{self.n_epochs_per_frame}: {str(self.losses)}, Loss: {loss.item():.4f} | Runtime: {end - start:.4f}s")
            #
            # # Wandb logging
            # if self.wandb_available:
            #     wandb.log({"Epoch": epoch, "Loss": loss.item(), "Runtime": end - start})

            # Main run logging
            if epoch > self.n_epochs_warmup and (epoch - self.n_epochs_warmup + 1) % self.n_epochs_per_frame == 0 and n_frames_current < self.n_frames:
                logger.info(f"<-- End frame {n_frames_current - 1} -->")
                logger.info(f"<-- Start frame {n_frames_current} -->")
                n_frames_current += 1
                curr_loss = -1.0
                x_pred = torch.cat([x_pred, x1.detach()], dim=0)

        opt_end = time.time()
        logger.info(f"Optimization runtime: {opt_end - opt_start:.4f}s, {(opt_end - opt_start)/60:.2f}min")
        # Save outputs
        self.save_outputs(x_pred=x_pred, output_path=output_path, surfaces_name=surfaces_name, rgbs_name=rgbs_name, blurred_masks_name=blurred_masks_name, materials=materials, **kwargs)

        # Evaluation
        sequence_error = self.evaluate(x_pred=x_pred)

        return sequence_error

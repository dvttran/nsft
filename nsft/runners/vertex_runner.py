from .base_runner import BaseRunner, wandb_available
import time
from loguru import logger
if wandb_available:
    import wandb
import torch
from typing import Union
from pathlib import Path


class VertexRunner(BaseRunner, name="Vertex"):
    def __init__(self, n_epochs_warmup: int = 0, n_epochs_per_frame: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.n_epochs_warmup = n_epochs_warmup
        self.n_epochs_per_frame = n_epochs_per_frame
        self.n_frames = self.dataset.n_time_span
        self.n_epochs = n_epochs_warmup + n_epochs_per_frame * self.n_frames

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
        x_pred = self.x0[None].clone()
        x1 = x0[None].clone()
        x1.requires_grad = True
        self.optimizer = self.get_optimizer([x1])
        n_frames_current = 1

        opt_start = time.time()
        for epoch in range(self.n_epochs):
            start = time.time()

            self.optimizer.zero_grad()
            # Loss
            loss = self.step(x1, n_frames_current)

            # Back propagation
            loss.backward()
            self.optimizer.step()

            end = time.time()

            # Meta data Warmup logging
            if epoch == 0 and self.n_epochs_warmup > 0:
                logger.info(f"<-- Warmup {self.n_epochs_warmup} epochs starting ... -->")
                logger.info(f"<-- Start frame {n_frames_current - 1} -->")
            if epoch == self.n_epochs_warmup:
                if self.n_epochs_warmup > 0:
                    logger.info(f"<-- Warmup end ! -->")
                logger.info(f"<-- Main run {self.n_epochs_per_frame} epochs/frame starting ... -->")

            # Warm up logging
            if epoch < self.n_epochs_warmup and (epoch == 0 or (epoch + 1) % self.log_every == 0 or epoch == self.n_epochs_warmup - 1):
                logger.info(f"Warmup {epoch + 1}/{self.n_epochs_warmup}: {str(self.losses)}, Loss: {loss.item():.4f} | Runtime: {end - start:.4f}s")

            # Main run logging
            if epoch >= self.n_epochs_warmup and ((epoch - self.n_epochs_warmup) % self.n_epochs_per_frame == 0 or (epoch + 1) % self.log_every == 0 or (epoch - self.n_epochs_warmup) % self.n_epochs_per_frame == self.n_epochs_per_frame - 1):
                logger.info(f"Epoch {((epoch - self.n_epochs_warmup)  % (self.n_epochs_per_frame) + 1)}/{self.n_epochs_per_frame}: {str(self.losses)}, Loss: {loss.item():.4f} | Runtime: {end - start:.4f}s")

            # Wandb logging
            if self.wandb_available:
                wandb.log({"Epoch": epoch, "Loss": loss.item(), "Runtime": end - start})

            # Main run logging
            if epoch > self.n_epochs_warmup and (epoch - self.n_epochs_warmup + 1) % self.n_epochs_per_frame == 0 and n_frames_current < self.n_frames:
                logger.info(f"<-- End frame {n_frames_current - 1} -->")
                logger.info(f"<-- Start frame {n_frames_current} -->")
                n_frames_current += 1
                x_pred = torch.cat([x_pred, x1.detach()], dim=0)

        opt_end = time.time()
        logger.info(f"Optimization runtime: {opt_end - opt_start:.4f}s, {(opt_end - opt_start)/60:.2f}min")
        # Save outputs
        self.save_outputs(x_pred=x_pred, output_path=output_path, surfaces_name=surfaces_name, rgbs_name=rgbs_name, blurred_masks_name=blurred_masks_name, materials=materials, **kwargs)

        # Evaluation
        sequence_error = self.evaluate(x_pred=x_pred)

        return sequence_error

from .base_runner import BaseRunner, wandb_available
import time
from loguru import logger
if wandb_available:
    import wandb
import torch
from typing import Union
from pathlib import Path


class WindowRunner(BaseRunner, name="Window"):
    def __init__(self, n_epochs_warmup: int = 0, n_epochs_per_window: int = 500, **kwargs):
        super().__init__(**kwargs)
        self.window_size = 3  # TODO: modify later
        self.n_epochs_warmup = n_epochs_warmup
        self.n_epochs_per_window = n_epochs_per_window
        self.n_frames = self.dataset.n_time_span
        self.n_windows = self.n_frames - self.window_size + 1  # start_idx = 0, end_idx = n_frames - self.window_size
        self.n_epochs = n_epochs_warmup + n_epochs_per_window * self.n_windows
        self.time_span = torch.linspace(0, 1, self.window_size).to(self.device)

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
        n_windows_current = 1
        warmup = True

        opt_start = time.time()
        for epoch in range(self.n_epochs):
            start = time.time()

            self.optimizer.zero_grad()
            # x1 = (x0 + self.model(x0))[None]
            x_window = self.model(x0, self.time_span, rtol=1e-7, atol=1e-9, method="rk4")

            loss, temporal_loss = self.window_step(x_window, n_windows_current, warmup, self.window_size)

            # Back propagation
            loss = loss + temporal_loss
            loss.backward()
            self.optimizer.step()

            end = time.time()

            # Meta data Warmup logging
            if epoch == 0 and self.n_epochs_warmup > 0:
                logger.info(f"<-- Warmup {self.n_epochs_warmup} epochs starting ... -->")
                logger.info(f"<-- Start window {n_windows_current - 1} -->")
            if epoch == self.n_epochs_warmup:
                if self.n_epochs_warmup > 0:
                    logger.info(f"<-- Warmup end ! -->")
                    warmup = False
                logger.info(f"<-- Main run {self.n_epochs_per_window} epochs/window starting ... -->")

            # Warm up logging
            if epoch < self.n_epochs_warmup and (epoch == 0 or (epoch + 1) % self.log_every == 0 or epoch == self.n_epochs_warmup - 1):
                logger.info(f"Warmup {epoch + 1}/{self.n_epochs_warmup}: {str(self.losses)}, L_temp: {temporal_loss.item():.4f}, Loss: {loss.item():.4f} | Runtime: {end - start:.4f}s")

            # Main run logging
            if epoch >= self.n_epochs_warmup and ((epoch - self.n_epochs_warmup) % self.n_epochs_per_window == 0 or (epoch + 1) % self.log_every == 0 or (epoch - self.n_epochs_warmup) % self.n_epochs_per_window == self.n_epochs_per_window - 1):
                logger.info(f"Epoch {((epoch - self.n_epochs_warmup)  % (self.n_epochs_per_window) + 1)}/{self.n_epochs_per_window}: {str(self.losses)}, L_temp: {temporal_loss.item():.4f}, Loss: {loss.item():.4f} | Runtime: {end - start:.4f}s")

            # Wandb logging
            if self.wandb_available:
                wandb.log({"Epoch": epoch, "Loss": loss.item(), "Runtime": end - start})

            # Main run logging
            if epoch > self.n_epochs_warmup and (epoch - self.n_epochs_warmup + 1) % self.n_epochs_per_window == 0 and n_windows_current < self.n_windows:
                logger.info(f"<-- End window {n_windows_current - 1} -->")
                logger.info(f"<-- Start window {n_windows_current} -->")
                n_windows_current += 1
                x0 = x_window[1].detach()
                x_pred = torch.cat([x_pred, x0[None]], dim=0)

            # last epoch
            if epoch == self.n_epochs - 1:
                x_pred = torch.cat([x_pred, x_window[1:].detach()], dim=0)

        opt_end = time.time()
        logger.info(f"Optimization runtime: {opt_end - opt_start:.4f}s, {(opt_end - opt_start)/60:.2f}min")
        # Save outputs
        self.save_outputs(x_pred=x_pred, output_path=output_path, surfaces_name=surfaces_name, rgbs_name=rgbs_name, blurred_masks_name=blurred_masks_name, materials=materials, **kwargs)

        # Evaluation
        sequence_error = self.evaluate(x_pred=x_pred)

        return sequence_error

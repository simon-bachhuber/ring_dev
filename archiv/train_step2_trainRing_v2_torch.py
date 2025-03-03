"""
#!/bin/bash -l
#SBATCH --gres=gpu:l40s:4
#SBATCH --time=60:00:00
#SBATCH --partition=imes
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=15G
"""

# flake8: noqa E402

import os

os.environ["JAX_PLATFORM_NAME"] = "cpu"

from absl import flags
from diodem.benchmark import benchmark
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

torch.set_float32_matmul_precision("high")

from lightning.pytorch.loggers import WandbLogger
import numpy as np
import tree

from load_dl_v2 import load_ds_train_ds_val
from load_dl_v2 import load_imtp
import torch_math


class MotionTransformer(nn.Module):
    def __init__(
        self,
        num_segments,
        input_dim,
        d_model=128,
        nhead=4,
        num_layers=4,
        epsilon=1e-6,
        encoder_only: bool = False,
    ):
        super(MotionTransformer, self).__init__()
        self.num_segments = num_segments
        self.input_dim = num_segments * input_dim
        self.output_dim = num_segments * 4
        self.epsilon = epsilon  # Small constant for stability
        self.encoder_only = encoder_only

        if encoder_only:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=num_layers
            )
        else:
            # Transformer layers
            self.transformer = nn.Transformer(
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_layers,
                num_decoder_layers=num_layers,
                batch_first=True,
            )

        # Linear layers for input projection and output decoding
        self.input_proj = nn.Linear(self.input_dim, d_model)
        self.output_proj = nn.Linear(d_model, self.output_dim)

    def forward(self, x):
        # Reshape input to (T, batch_size, input_dim)
        batch_size, T, N, _ = x.shape
        x = x.view(batch_size, T, -1)  # (batch_size, T, N*input_dim)

        # Pass through the Transformer
        x = self.input_proj(x)  # Project to model dimension
        x = (x,) if self.encoder_only else (x, x)
        x = self.transformer(*x)  # Transformer (T, batch_size, d_model)
        x = self.output_proj(x)  # Decode to output dimension

        # Reshape back to (batch_size, T, N, 4)
        x = x.view(batch_size, T, self.num_segments, 4)

        # Normalize quaternions to ensure unit norm with stability
        norm = torch.norm(x, dim=-1, keepdim=True)  # Compute norm
        x = x / (norm + self.epsilon)  # Add epsilon to prevent division by zero
        return x


class ToFloat32Transform:
    def __init__(self, ds):
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        return tree.map_structure(lambda a: a.astype(np.float32), self.ds[idx])


def _diodem_dl(segments: list[str], exp_id, motion_start) -> DataLoader:
    imtp = load_imtp().replace(segments=segments)
    X, y, *_ = benchmark(imtp, exp_id=exp_id, motion_start=motion_start)

    class _Dataset:
        def __len__(self):
            return 1

        def __getitem__(self, idx):
            return X, y

    dl = DataLoader(_Dataset())
    dl._diodem_name = imtp.name(exp_id, motion_start)
    return dl


flags.DEFINE_integer("rnn_w", 800, "hidden size of RNN cell")
flags.DEFINE_integer("rnn_d", 3, "number of RNN cells")
FLAGS = flags.FLAGS
flags.FLAGS(
    ["myname", "--path_lam4=/bigwork/nhkbbach/data2/v2_lam4_kin_rigid", "--four_seg"]
)


class Transformer(L.LightningModule):
    def __init__(
        self,
        truncated_bptt_steps=None,
        encoder_only=False,
        model_dim=512,
        T_max: int = 100,
        nhead=16,
        num_layers=6,
    ):
        super().__init__()
        input_dim = load_imtp().getF()
        self.net = MotionTransformer(
            4,
            input_dim,
            model_dim,
            encoder_only=encoder_only,
            nhead=nhead,
            num_layers=num_layers,
        )
        self.truncated_bptt_steps = truncated_bptt_steps

        if truncated_bptt_steps is not None:
            # 1. Switch to manual optimization
            self.automatic_optimization = False

        self.T_max = T_max

    # 2. Remove the `hiddens` argument
    def training_step(self, batch, batch_idx):
        X, Y = batch

        if self.truncated_bptt_steps is not None:
            opt = self.optimizers()
            scheduler = self.lr_schedulers()

            # 3. Split the batch in chunks along the time dimension
            T = X.shape[1]
            hiddens = None
            losses = []
            for start in range(0, T, self.truncated_bptt_steps):
                split = slice(start, start + self.truncated_bptt_steps)
                x, y = X[:, split, ...], Y[:, split, ...]
                yhat, hiddens = self.forward(x, hiddens)
                loss = self.criterion(y, yhat)

                opt.zero_grad()
                self.manual_backward(loss)
                torch.nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                opt.step()

                # 5. "Truncate"
                hiddens = tree.map_structure(lambda t: t.detach(), hiddens)

                losses.append(loss)

            train_loss = torch.stack(losses).mean()
            scheduler.step(metrics=train_loss)
            self.log(
                "train/loss",
                train_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )
            # 6. Remove the return of `hiddens`
            # Returning loss in manual optimization is not needed
            return None
        else:
            yhat, _ = self(X)
            loss = self.criterion(Y, yhat)
            self.log(
                "train/loss",
                loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            return loss

    def validation_step(self, batch, _):
        x, y = batch
        yhat, _ = self(x)
        val_loss = self.criterion(y, yhat)
        self.log(
            "val/loss",
            val_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def forward(self, x):
        return self.net(x), None

    def forward_rnn(self, x, hiddens=None):
        output, hiddens = self.lstm(x, hiddens)
        qs = self.linear(output)
        for i in range(qs.shape[-1], 4):
            qs[..., i : (i + 4)] = torch_math.safe_normalize(qs[..., i : (i + 4)])
        return qs, hiddens

    def init_hidden(self, bs: int):
        h0 = torch.zeros((FLAGS.rnn_d, bs, FLAGS.rnn_w))
        c0 = torch.zeros((FLAGS.rnn_d, bs, FLAGS.rnn_w))
        return (h0, c0)

    def criterion(self, q, qhat):
        return (
            torch_math.loss_fn([-1, 0, 1, 2], q[:, -100:], qhat[:, -100:])
            .square()
            .mean()
        )

    def criterion_rnn(self, q, qhat):
        T = q.shape[-3]
        return (
            torch_math.loss_fn(
                [-1, 0, 1, 2], q.reshape(-1, T, 4, 4), qhat.reshape(-1, T, 4, 4)
            )
            .square()
            .mean()
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=self.T_max
                ),
                "monitor": "train/loss",
            },
        }


if __name__ == "__main__":
    rnn = Transformer(
        encoder_only=True,
        model_dim=512,
        T_max=600,
        nhead=8,
        num_layers=4,
    )
    rnn = torch.compile(rnn)
    ds_train, ds_val = load_ds_train_ds_val(rnno=True, flatten=False, T=2000)
    num_workers = 4
    dl_train = DataLoader(
        ToFloat32Transform(ds_train),
        64,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )
    dl_val = DataLoader(
        ToFloat32Transform(ds_val),
        64,
        drop_last=True,
        num_workers=num_workers,
        multiprocessing_context="spawn",
        persistent_workers=True,
    )

    logger = WandbLogger(name="bi-gru", project="imt-torch")
    logger.watch(rnn)
    trainer = L.Trainer(
        devices=4,
        max_epochs=rnn.T_max,
        strategy="ddp",
        callbacks=[LearningRateMonitor("step")],
        logger=logger,
        gradient_clip_val=0.5,
        accumulate_grad_batches=1,
    )
    trainer.fit(rnn, dl_train, dl_val)

import itertools
import math
from typing import Optional, Union, Sequence, Tuple

import numpy as np
import torch
from einops import rearrange
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import Linear, MSELoss, LeakyReLU
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import LambdaLR

from marlin_pytorch.model.decoder import MarlinDecoder
from marlin_pytorch.model.encoder import MarlinEncoder
from marlin_pytorch.model.modules import MLP


class Marlin(LightningModule):

    def __init__(self,
        img_size=224,
        patch_size=16,
        n_frames=16,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=8,
        mlp_ratio=4.,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        norm_layer="LayerNorm",
        init_values=0.,
        tubelet_size=2,
        optimizer_type: str = "AdamW",
        optimizer_eps: float = 1e-8,
        optimizer_betas: Tuple[float, float] = (0.9, 0.999),
        weight_decay: float = 0.,
        learning_rate: float = 1.5e-4,
        warmup_lr: float = 1e-6,
        min_lr: float = 1e-5,
        warmup_epochs: int = 40,
        max_epochs: int = 2000,
        iter_per_epoch: int = 1297,
        distributed: bool = False,
        d_steps: int = 3,
        g_steps: int = 1,
        adv_weight: float = 0.1,
        gp_weight: float = 10.,
        name: str = None
    ):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = MarlinEncoder(
            img_size=img_size,
            patch_size=patch_size,
            n_frames=n_frames,
            embed_dim=encoder_embed_dim,
            depth=encoder_depth,
            num_heads=encoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )

        self.decoder = MarlinDecoder(
            img_size=img_size,
            patch_size=patch_size,
            embed_dim=decoder_embed_dim,
            depth=decoder_depth,
            num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            norm_layer=norm_layer,
            init_values=init_values,
            tubelet_size=tubelet_size
        )

        self.discriminator = MLP([patch_size * patch_size * 3 * tubelet_size, int(decoder_embed_dim * mlp_ratio), 1],
                                 build_activation=LeakyReLU)

        self.tubelet_size = tubelet_size
        self.patch_size = patch_size

        if optimizer_type == "AdamW":
            self.optimizer_type = AdamW
        elif optimizer_type == "Adam":
            self.optimizer_type = Adam
        else:
            raise ValueError("optimizer_type must be either AdamW or Adam")

        self.optimizer_eps = optimizer_eps
        self.optimizer_betas = optimizer_betas
        self.weight_decay = weight_decay

        self.learning_rate = learning_rate
        self.warmup_lr = warmup_lr
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.iter_per_epoch = iter_per_epoch

        self.d_steps = d_steps
        self.g_steps = g_steps
        self.adv_weight = adv_weight
        self.gp_weight = gp_weight

        self.lr_scheduler_factors = self._cosine_scheduler_factors()

        self.enc_dec_proj = Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.loss_fn = MSELoss()
        self.automatic_optimization = False
        self.distributed = distributed
        self.name = name

    def forward(self, x, mask):
        x = self.encoder(x, mask)
        x = self.enc_dec_proj(x)
        x = self.decoder(x, mask)
        return x

    @staticmethod
    def g_loss_fn(pred) -> Tensor:
        return -pred.mean()

    @staticmethod
    def d_loss_fn(pred, target) -> Tensor:
        real_score = pred[target == 1]
        fake_score = pred[target == 0]
        return fake_score.mean() - real_score.mean()

    def gradient_penalty_fn(self, real_patches: Tensor, fake_patches: Tensor) -> Tensor:
        alpha = torch.rand(1).to(self.device).expand(real_patches.size())

        interpolates = torch.autograd.Variable(alpha * real_patches + ((1 - alpha) * fake_patches), requires_grad=True)
        disc_interpolates = self.discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                        grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                        create_graph=True)[0]

        gradients = rearrange(gradients, 'b n c -> (b n) c')
        gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gp

    def step(self, batch):
        # forward step
        x, mask = batch
        pred = self(x, mask)
        # get patches
        y = x.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        y = rearrange(y, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
        # filter masked patches
        b, _, c = y.shape
        y = y[~mask].view(b, -1, c)
        return pred, y

    def g_step(self, pred, y):
        # rec loss
        rec_loss = self.loss_fn(pred, y)

        # adversarial loss
        adv_loss = self.adv_weight * self.g_loss_fn(self.discriminator(pred)).mean()
        return {"loss": rec_loss + adv_loss, "g_loss": rec_loss + adv_loss, "rec_loss": rec_loss, "adv_loss": adv_loss}

    def d_step(self, pred, y):
        # forward discriminator
        fake_labels = torch.zeros(pred.size(0), pred.size(1), 1, device=self.device)  # fake labels are zeros
        real_labels = torch.ones(y.size(0), y.size(1), 1, device=self.device)  # real labels are ones
        assert fake_labels.shape == real_labels.shape
        d_batch = torch.cat((pred, y), dim=0)
        d_labels = torch.cat((fake_labels, real_labels), dim=0)
        d_loss = self.d_loss_fn(self.discriminator(d_batch), d_labels).mean()
        if self.training and self.gp_weight > 0:
            gp = self.gradient_penalty_fn(y, pred.detach())
            total_loss = d_loss + self.gp_weight * gp
            return {"loss": total_loss, "d_loss": total_loss, "d_loss0": d_loss, "gp": gp}
        else:
            return {"loss": d_loss, "d_loss": d_loss, "d_loss0": d_loss}

    def training_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
    ) -> Tensor:
        g_optimizer, d_optimizer = self.optimizers()
        schedulers = self.lr_schedulers()
        if schedulers is not None:
            g_scheduler, d_scheduler = schedulers
        else:
            g_scheduler, d_scheduler = None, None

        # train discriminator
        d_loss = None
        d_result = None
        for _ in range(self.d_steps):
            d_optimizer.zero_grad()
            pred, y = self.step(batch)
            d_result = self.d_step(pred, y)
            d_loss = d_result["loss"]
            self.manual_backward(d_loss)
            d_optimizer.step()
        if d_scheduler is not None and batch_idx == 0:
            d_scheduler.step()

        # train generator
        g_loss = None
        g_result = None
        for _ in range(self.g_steps):
            g_optimizer.zero_grad()
            pred, y = self.step(batch)
            g_result = self.g_step(pred, y)
            g_loss = g_result["loss"]
            self.manual_backward(g_loss)
            g_optimizer.step()
        if g_scheduler is not None and batch_idx == 0:
            g_scheduler.step()

        loss_dict = {
            "loss": d_loss + g_loss,
            **{k: v for k, v in d_result.items() if k != "loss"},
            **{k: v for k, v in g_result.items() if k != "loss"},
        }
        self.log_dict({f"train_{k}": v for k, v in loss_dict.items()}, on_step=True, on_epoch=True,
            prog_bar=False, sync_dist=self.distributed)

        return loss_dict["loss"]

    def validation_step(self, batch: Optional[Union[Tensor, Sequence[Tensor]]] = None, batch_idx: Optional[int] = None,
        dataloader_idx: Optional[int] = None
    ) -> Tensor:
        if batch_idx == 0:
            self._log_sample_reconstruction_image(batch)
        pred, y = self.step(batch)

        # test discriminator
        d_result = self.d_step(pred, y)
        d_loss = d_result["loss"]

        # test generator
        g_result = self.g_step(pred, y)
        g_loss = g_result["loss"]

        loss_dict = {
            "loss": d_loss + g_loss,
            **{k: v for k, v in d_result.items() if k != "loss"},
            **{k: v for k, v in g_result.items() if k != "loss"},
        }
        self.log_dict({f"val_{k}": v for k, v in loss_dict.items()}, on_step=False, on_epoch=True,
            prog_bar=True, sync_dist=self.distributed)
        return loss_dict["loss"]

    def _log_sample_reconstruction_image(self, batch):
        x, mask = batch
        x = x[:1]
        mask = mask[:1]
        y = self(x, mask)
        # make gt image
        gt_img = x.unfold(2, self.tubelet_size, self.tubelet_size) \
            .unfold(3, self.patch_size, self.patch_size) \
            .unfold(4, self.patch_size, self.patch_size)
        gt_img = rearrange(gt_img, "b c nt nh nw pt ph pw -> b (nt nh nw) (c pt ph pw)")
        gt_img = self.decoder.unpatch_to_img(gt_img).detach()[0, :, 0]  # (C, H, W)
        # make rec image
        # patch x
        x = rearrange(x, "b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c",
            p0=self.tubelet_size, p1=self.patch_size, p2=self.patch_size)
        x = rearrange(x, "b n p c -> b n (c p)")
        x_rec = x.clone()
        x_rec[~mask] = y.view(-1, self.patch_size * self.patch_size * self.tubelet_size * 3)
        rec_img = self.decoder.unpatch_to_img(x_rec).detach()[0, :, 0]  # (C, H, W)
        # make masked original image
        x_masked = x.clone()
        x_masked[~mask] = 0.5
        masked_img = self.decoder.unpatch_to_img(x_masked).detach()[0, :, 0]  # (C, H, W)

        # log images
        log_img = torch.cat([gt_img, masked_img, rec_img], dim=2)
        self.log_image("sample", log_img)

    def _cosine_scheduler_factors(self):
        warmup_schedule = np.array([])
        warmup_iters = self.warmup_epochs * self.iter_per_epoch
        if self.warmup_epochs > 0:
            warmup_schedule = np.linspace(0, self.learning_rate, warmup_iters)

        iters = np.arange(self.max_epochs * self.iter_per_epoch - warmup_iters)
        schedule = np.array(
            [self.min_lr + 0.5 * (self.learning_rate - self.min_lr) * (1 + math.cos(math.pi * i / (len(iters))))
                for i in iters])

        schedule = np.concatenate((warmup_schedule, schedule))

        assert len(schedule) == self.max_epochs * self.iter_per_epoch
        values_factors = schedule[::self.iter_per_epoch] / self.learning_rate
        return values_factors

    def _cosine_scheduler_fn(self, epoch):
        return self.lr_scheduler_factors[epoch]

    def configure_optimizers(self):
        g_optimizer = self.optimizer_type(
            itertools.chain(self.encoder.parameters(), self.decoder.parameters(), self.enc_dec_proj.parameters()),
            lr=self.learning_rate,
            eps=self.optimizer_eps,
            betas=self.optimizer_betas,
            weight_decay=self.weight_decay)

        g_lr_scheduler = LambdaLR(
            g_optimizer,
            lr_lambda=self._cosine_scheduler_fn
        )

        d_optimizer = self.optimizer_type(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            eps=self.optimizer_eps,
            betas=self.optimizer_betas,
            weight_decay=self.weight_decay)

        d_lr_scheduler = LambdaLR(
            d_optimizer,
            lr_lambda=self._cosine_scheduler_fn
        )

        return [g_optimizer, d_optimizer], [g_lr_scheduler, d_lr_scheduler]

    def log_image(self, name: str, image: torch.Tensor) -> None:
        """Log an image to the logger"""
        if self.logger is None:
            return

        self.logger.experiment.add_image(name, torch.clip(image, 0, 1), self.trainer.global_step)

import torch
from torch import nn
from torch.nn import functional as F
import lightning as L
from transformers.optimization import get_linear_schedule_with_warmup

from .mamba_bidirectional import MambaBlock
import math

# convert image features into video features

# copied from https://github.com/helboukkouri/character-bert/blob/main/modeling/character_cnn.py
class Highway(torch.nn.Module):
    """
    A [Highway layer](https://arxiv.org/abs/1505.00387) does a gated combination of a linear
    transformation and a non-linear transformation of its input.  :math:`y = g * x + (1 - g) *
    f(A(x))`, where :math:`A` is a linear transformation, :math:`f` is an element-wise
    non-linearity, and :math:`g` is an element-wise gate, computed as :math:`sigmoid(B(x))`.

    This module will apply a fixed number of highway layers to its input, returning the final
    result.

    # Parameters

    input_dim : `int`, required
        The dimensionality of :math:`x`.  We assume the input has shape `(batch_size, ...,
        input_dim)`.
    num_layers : `int`, optional (default=`1`)
        The number of highway layers to apply to the input.
    activation : `Callable[[torch.Tensor], torch.Tensor]`, optional (default=`torch.nn.functional.relu`)
        The non-linearity to use in the highway layers.
    """

    def __init__(
        self,
        input_dim: int,
        num_layers: int = 1,
        activation=torch.nn.functional.relu,
    ):
        super().__init__()
        self._input_dim = input_dim
        self._layers = torch.nn.ModuleList(
            [torch.nn.Linear(input_dim, input_dim * 2) for _ in range(num_layers)]
        )
        self._activation = activation
        for layer in self._layers:
            # We should bias the highway layer to just carry its input forward.  We do that by
            # setting the bias on `B(x)` to be positive, because that means `g` will be biased to
            # be high, so we will carry the input forward.  The bias on `B(x)` is the second half
            # of the bias vector in each Linear layer.
            layer.bias[input_dim:].data.fill_(1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        current_input = inputs
        for layer in self._layers:
            projected_input = layer(current_input)
            linear_part = current_input
            # NOTE: if you modify this, think about whether you should modify the initialization
            # above, too.
            nonlinear_part, gate = projected_input.chunk(2, dim=-1)
            nonlinear_part = self._activation(nonlinear_part)
            gate = torch.sigmoid(gate)
            current_input = gate * linear_part + (1 - gate) * nonlinear_part
        return current_input


class VIDEO_QUERY_REP_PROJECTION(nn.Module):
    def __init__(
        self, img_query_rep_dim=512, video_query_rep_dim=768, hidden_size=None
    ):
        super(VIDEO_QUERY_REP_PROJECTION, self).__init__()

        self.img_query_rep_dim = img_query_rep_dim
        self.video_query_rep_dim = video_query_rep_dim

        if not hidden_size:
            hidden_size = img_query_rep_dim

        # TODO: add dropout
        # self.proj = nn.Sequential(
        #     nn.Linear(img_query_rep_dim, video_query_rep_dim),
        #     nn.GELU(),
        #     # nn.Linear(hidden_size, video_query_rep_dim),
        #     # nn.ReLU(),
        #     Highway(video_query_rep_dim, 3)
        # )

        self.proj = nn.Sequential(
            Highway(img_query_rep_dim, 4),
            nn.GELU(),
            nn.Linear(img_query_rep_dim, video_query_rep_dim),
            # nn.Linear(hidden_size, video_query_rep_dim),
            # nn.ReLU(),
        )

    def forward(self, img_query_rep):
        # input: (*, input_dim)

        video_query_rep = self.proj(img_query_rep)

        return video_query_rep


class VIDEO_MIXER(nn.Module):
    def __init__(
        self, img_rep_dim=512, video_rep_dim=768, n_layer=12, bidirectional=False
    ):
        super(VIDEO_MIXER, self).__init__()

        self.f_proj = nn.Sequential(nn.Linear(img_rep_dim, video_rep_dim), nn.ReLU())
        self.mixer = MambaBlock(
            d_model=video_rep_dim, n_layer=n_layer, bidirectional=bidirectional
        )

    def forward(self, frames_rep, seq_lens=None):
        # input: (batch, num_frames, img_rep_dim)

        pframes_rep = self.f_proj(frames_rep)

        video_rep = self.mixer(pframes_rep, seq_lens=seq_lens)

        return video_rep


def contrastive_loss(anchor, positive, negative, temperature):
    # input vectors should be normalized
    # A, anchor: (seq_len, dim)
    # P, positive: (seq_len, dim)
    # N, negative: (seq_len2, dim)
    # 1 positive instance and seq_len2 negative instances for each anchor

    device = anchor.device

    seq_len, dim = anchor.shape
    seq_len2, _ = negative.shape

    # (seq_len)
    # AP_similarity = anchor @ positive.T
    AP_similarity = torch.bmm(anchor.unsqueeze(1), positive.unsqueeze(2))
    # (seq_len, seq_len2)
    AN_similarity = anchor @ negative.T

    # (seq_len,)
    numerator = torch.exp(AP_similarity / temperature)

    # (seq_len,)
    denominator = torch.exp(AN_similarity / temperature).sum(-1)

    loss = -torch.log(numerator / denominator).mean()

    return loss


class VIDEO_CONTRASTIVE_LEARNING(L.LightningModule):
    def __init__(
        self,
        img_rep_dim=512,
        video_rep_dim=768,
        n_layer=6,
        bidirectional=False,
        peak_lr=3e-5,
        last_lr=1e-6,
        beta_1=0.9,
        beta_2=0.95,
        weight_decay=0.1,
        eps=1e-08,
        lr_warmup_perc=0.1,  # lr warmup for the first 10% of the training
        temperature=1,
        **other_args,
    ):
        super(VIDEO_CONTRASTIVE_LEARNING, self).__init__()
        # img_rep_dim == img_query_rep_dim
        # video_rep_dim == video_query_rep_dim

        self.peak_lr = peak_lr
        self.last_lr = last_lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.weight_decay = weight_decay
        self.eps = eps
        self.lr_warmup_perc = lr_warmup_perc

        self.temperature = temperature

        # convert text query into video query
        self.query_proj = VIDEO_QUERY_REP_PROJECTION(
            img_query_rep_dim=img_rep_dim,
            video_query_rep_dim=video_rep_dim,
            hidden_size=None,
        )

        # compute video representations from a list of frame's embedding
        self.video_mixer = VIDEO_MIXER(
            img_rep_dim=img_rep_dim,
            video_rep_dim=video_rep_dim,
            n_layer=n_layer,
            bidirectional=False,  # a frame embedding should only remember events that happend before it to reduce redundancy
        )

    def calc_loss(self, frames_rep, seq_lens, positive_queries, negative_queries):
        # frames_rep: (batch, max_number_frame, img_rep_dim)

        batch_size, _, _ = frames_rep.shape

        # (batch, max_number_frame, video_rep_dim)
        video_rep = self.video_mixer(frames_rep, seq_lens)

        video_rep = F.normalize(video_rep, dim=-1)

        loss = []

        for i in range(batch_size):

            # (seq_len, dim)
            positive_query = F.normalize(self.query_proj(positive_queries[i]), dim=-1)
            negative_query = F.normalize(self.query_proj(negative_queries[i]), dim=-1)

            seq_len = seq_lens[i]
            anchor_f = video_rep[i][:seq_len]

            loss.append(
                contrastive_loss(
                    anchor_f,
                    positive_query,
                    negative_query,
                    temperature=self.temperature,
                )
            )

        return torch.hstack(loss).mean()

    def training_step(self, batch):
        loss = self.calc_loss(
            batch["frames_rep"],
            batch["seq_lens"],
            positive_queries=batch["positive_queries"],
            negative_queries=batch["negative_queries"],
        )

        self.log("train_loss", loss.detach(), prog_bar=True)

        if torch.isnan(loss):
            raise Exception(f"Loss is NaN")

        return loss

    def validation_step(self, batch):
        loss = self.calc_loss(
            batch["frames_rep"],
            batch["seq_lens"],
            positive_queries=batch["positive_queries"],
            negative_queries=batch["negative_queries"],
        )

        self.log(
            "valid_loss",
            loss.detach(),
            prog_bar=True,
            batch_size=len(batch["frames_rep"]),
        )

    def num_steps(self) -> int:
        """Get number of steps"""
        # Accessing _data_source is flaky and might break
        dataset = self.trainer.fit_loop._data_source.dataloader()
        dataset_size = len(dataset)
        num_devices = max(1, self.trainer.num_devices)
        num_steps = (
            dataset_size
            * self.trainer.max_epochs
            // (self.trainer.accumulate_grad_batches * num_devices)
        )
        return num_steps

    def configure_optimizers(self):
        if self.trainer.max_epochs == -1:
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=self.peak_lr)

            return self.optimizer

        betas = (self.beta_1, self.beta_2)

        self.optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.peak_lr,
            weight_decay=self.weight_decay,
            betas=betas,
            eps=self.eps,
        )

        # num_steps = self.num_steps()
        # self.lr_scheduler = get_linear_schedule_with_warmup(
        #     self.optimizer,
        #     num_warmup_steps=int(num_steps * self.config.optimizer.warmup_perc),
        #     num_training_steps=num_steps,
        # )
        # return [self.optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

        def get_scheduler(
            optimizer, num_training_steps, warmup_steps, peak_lr, last_lr
        ):

            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return current_step / warmup_steps
                progress = (current_step - warmup_steps) / (
                    num_training_steps - warmup_steps
                )
                cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
                lr = last_lr + (peak_lr - last_lr) * cosine_decay
                return lr / peak_lr

            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        num_steps = self.num_steps()

        self.scheduler = get_scheduler(
            self.optimizer,
            num_steps,
            int(num_steps * self.lr_warmup_perc),
            self.peak_lr,
            self.last_lr,
        )

        lr_scheduler = {
            "scheduler": self.scheduler,
            "name": "custom_scheduler",
            "interval": "step",  # Ensure learning rate updates per step
            "frequency": 1,  # Optional: If you want to make sure it updates every step
        }

        return [self.optimizer], [lr_scheduler]

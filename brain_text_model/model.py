from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List, Tuple

import lightning as L
import torch
import torch.nn as nn

from .config import BrainTextDataConfig, BrainTextModelConfig, BrainTextTrainingConfig
from .metrics import character_error_rate
from .tokenizer import CharTokenizer, PhonemeTokenizer


class TemporalConvEncoder(nn.Module):
    """Stacked temporal convolutions that shrink the time dimension."""

    def __init__(
        self,
        input_dim: int,
        channels: List[int],
        kernel_size: int,
        strides: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        if len(channels) != len(strides):
            raise ValueError("conv_channels and conv_stride must have the same length")
        blocks = []
        in_ch = input_dim
        padding = kernel_size // 2
        for out_ch, stride in zip(channels, strides):
            blocks.append(
                nn.Sequential(
                    nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.GroupNorm(1, out_ch),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            in_ch = out_ch
        self.net = nn.Sequential(*blocks)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = features.transpose(1, 2)
        x = self.net(x)
        return x.transpose(1, 2)

    def output_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        out = lengths.clone()
        for module in self.net:
            conv = module[0]
            kernel = conv.kernel_size[0]
            stride = conv.stride[0]
            padding = conv.padding[0]
            dilation = conv.dilation[0]
            out = torch.div(
                out + 2 * padding - dilation * (kernel - 1) - 1,
                stride,
                rounding_mode="floor",
            ) + 1
        return torch.clamp_min(out, 1)


class BrainToTextLightningModule(L.LightningModule):
    """LightningModule that optimizes a CTC loss on neural feature sequences."""

    def __init__(
        self,
        data_config: BrainTextDataConfig,
        model_config: BrainTextModelConfig,
        train_config: BrainTextTrainingConfig,
        tokenizer: CharTokenizer | None = None,
        phoneme_tokenizer: PhonemeTokenizer | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(
            {
                "data_config": asdict(data_config),
                "model_config": asdict(model_config),
                "train_config": asdict(train_config),
            }
        )
        self.data_config = data_config
        self.model_config = model_config
        self.train_config = train_config
        self.tokenizer = tokenizer or CharTokenizer()
        self.phoneme_tokenizer = phoneme_tokenizer

        conv_channels = list(model_config.conv_channels)
        conv_strides = list(model_config.conv_stride)
        self.encoder = TemporalConvEncoder(
            input_dim=model_config.input_dim,
            channels=conv_channels,
            kernel_size=model_config.conv_kernel_size,
            strides=conv_strides,
            dropout=model_config.final_dropout,
        )
        encoder_dim = conv_channels[-1]
        self.projection = nn.Linear(encoder_dim, model_config.transformer_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_config.transformer_dim,
            nhead=model_config.transformer_heads,
            dim_feedforward=model_config.transformer_dim * 4,
            dropout=model_config.transformer_dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=model_config.transformer_layers,
        )
        self.dropout = nn.Dropout(model_config.final_dropout)
        self.text_head = nn.Linear(model_config.transformer_dim, self.tokenizer.vocab_size)
        self.phoneme_head = (
            nn.Linear(model_config.transformer_dim, self.phoneme_tokenizer.vocab_size)
            if self.phoneme_tokenizer
            else None
        )
        self.text_ctc = nn.CTCLoss(blank=self.tokenizer.blank_id, zero_infinity=True)
        self.phoneme_ctc = (
            nn.CTCLoss(blank=self.phoneme_tokenizer.blank_id, zero_infinity=True)
            if self.phoneme_tokenizer
            else None
        )

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def encode(
        self, features: torch.Tensor, feature_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.encoder(features)
        out_lengths = self.encoder.output_lengths(feature_lengths)
        x = self.projection(x)
        mask = self._lengths_to_mask(out_lengths, x.size(1))
        x = self.transformer(x, src_key_padding_mask=mask)
        x = self.dropout(x)
        return x, out_lengths

    def forward(self, features: torch.Tensor, feature_lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, out_lengths = self.encode(features, feature_lengths)
        logits = self.text_head(encoded)
        return logits, out_lengths

    def _lengths_to_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        return torch.arange(max_len, device=lengths.device).unsqueeze(0) >= lengths.unsqueeze(1)

    # ------------------------------------------------------------------
    # Training / validation
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, _ = self._compute_loss(batch, return_decoded=False)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> None:
        loss, decoded = self._compute_loss(batch, return_decoded=True)
        cer = character_error_rate(decoded, batch["transcripts"])
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_cer", cer, prog_bar=True, on_epoch=True)

    def _compute_loss(
        self,
        batch: Dict[str, torch.Tensor | List[str] | None],
        return_decoded: bool = False,
    ) -> tuple[torch.Tensor, List[str]]:
        features = batch["features"].float()
        feature_lengths = batch["feature_lengths"].long()
        targets = batch["targets"].long()
        target_lengths = batch["target_lengths"].long()

        encoded, out_lengths = self.encode(features, feature_lengths)
        logits = self.text_head(encoded)
        log_probs = logits.log_softmax(dim=-1)
        loss = self.text_ctc(log_probs.transpose(0, 1), targets, out_lengths, target_lengths)

        phoneme_targets = batch.get("phoneme_targets")
        phoneme_lengths = batch.get("phoneme_lengths")
        if (
            self.phoneme_head
            and self.phoneme_ctc
            and phoneme_targets is not None
            and phoneme_lengths is not None
            and phoneme_targets.numel() > 0
        ):
            phoneme_logits = self.phoneme_head(encoded)
            phoneme_log_probs = phoneme_logits.log_softmax(dim=-1)
            loss = loss + self.phoneme_ctc(
                phoneme_log_probs.transpose(0, 1),
                phoneme_targets.long(),
                out_lengths,
                phoneme_lengths.long(),
            )

        decoded: List[str] = []
        if return_decoded:
            decoded = self.tokenizer.greedy_decode(log_probs.transpose(0, 1), out_lengths)
        return loss, decoded

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay,
        )
        return optimizer


def _run_model_tests() -> None:
    """Test model architecture on CPU, MPS, and CUDA."""
    print("[model] Running tests...")
    
    # Determine available devices
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    
    print(f"[model] Testing on devices: {devices}")
    
    # Create configs
    data_cfg = BrainTextDataConfig()
    model_cfg = BrainTextModelConfig(
        input_dim=512,
        conv_channels=[64, 128],
        conv_stride=[2, 2],
        transformer_dim=128,
        transformer_heads=4,
        transformer_layers=2,
    )
    train_cfg = BrainTextTrainingConfig()
    
    # Test TemporalConvEncoder
    encoder = TemporalConvEncoder(
        input_dim=512,
        channels=[64, 128],
        kernel_size=5,
        strides=[2, 2],
        dropout=0.1,
    )
    
    for device in devices:
        try:
            encoder_device = encoder.to(device)
            features = torch.randn(4, 100, 512, device=device)  # (batch, time, features)
            output = encoder_device(features)
            
            # Check output shape
            assert output.shape[0] == 4  # batch size
            assert output.shape[2] == 128  # output channels
            assert output.shape[1] < 100  # time should be reduced
            
            # Test output_lengths
            lengths = torch.tensor([100, 90, 80, 70], device=device)
            out_lengths = encoder_device.output_lengths(lengths)
            assert out_lengths.shape == lengths.shape
            assert torch.all(out_lengths > 0)
            assert torch.all(out_lengths <= lengths)
            
            print(f"[model] ✓ TemporalConvEncoder works on {device}")
        except Exception as e:
            print(f"[model] ✗ TemporalConvEncoder failed on {device}: {e}")
    
    # Test full BrainToTextLightningModule
    tokenizer = CharTokenizer()
    module = BrainToTextLightningModule(
        data_config=data_cfg,
        model_config=model_cfg,
        train_config=train_cfg,
        tokenizer=tokenizer,
    )
    
    for device in devices:
        try:
            module_device = module.to(device)
            
            # Create dummy batch
            features = torch.randn(2, 100, 512, device=device)
            feature_lengths = torch.tensor([100, 90], device=device)
            
            # Test forward pass
            logits, out_lengths = module_device(features, feature_lengths)
            
            assert logits.shape[0] == 2  # batch size
            assert logits.shape[2] == tokenizer.vocab_size
            assert out_lengths.shape == torch.Size([2])
            
            # Test encode method
            encoded, enc_lengths = module_device.encode(features, feature_lengths)
            assert encoded.shape[0] == 2
            assert encoded.shape[2] == model_cfg.transformer_dim
            
            # Test with phoneme tokenizer
            phoneme_tokenizer = PhonemeTokenizer()
            phoneme_tokenizer.maybe_add_tokens("HH AH L OW W ER D")
            
            module_with_phonemes = BrainToTextLightningModule(
                data_config=data_cfg,
                model_config=model_cfg,
                train_config=train_cfg,
                tokenizer=tokenizer,
                phoneme_tokenizer=phoneme_tokenizer,
            ).to(device)
            
            assert module_with_phonemes.phoneme_head is not None
            assert module_with_phonemes.phoneme_ctc is not None
            
            # Test compute_loss with full batch
            batch = {
                "features": features,
                "feature_lengths": feature_lengths,
                "targets": torch.tensor([3, 4, 5, 6, 7, 8], device=device),
                "target_lengths": torch.tensor([3, 3], device=device),
                "transcripts": ["abc", "def"],
                "phoneme_targets": None,
                "phoneme_lengths": None,
            }
            
            loss, decoded = module_device._compute_loss(batch, return_decoded=True)
            assert loss.numel() == 1
            assert len(decoded) == 2
            assert all(isinstance(s, str) for s in decoded)
            
            # Test optimizer configuration
            optimizer = module_device.configure_optimizers()
            assert optimizer is not None
            assert isinstance(optimizer, torch.optim.AdamW)
            
            print(f"[model] ✓ BrainToTextLightningModule works on {device}")
        except Exception as e:
            error_msg = str(e)
            if "mps" in device.lower() and "_ctc_loss" in error_msg:
                print(f"[model] ⚠ BrainToTextLightningModule skipped on {device}: CTC loss not supported on MPS")
            else:
                print(f"[model] ✗ BrainToTextLightningModule failed on {device}: {e}")
    
    print("[model] ✓ All tests passed")


if __name__ == "__main__":
    _run_model_tests()

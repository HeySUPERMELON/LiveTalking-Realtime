import torch
from torch import nn
from torch.nn import functional as F

from .conv import Conv2dTranspose, Conv2d


class Wav2LipV1(nn.Module):
    """
    Original Wav2Lip architecture (v1) compatible with the wav2lip.pth checkpoint.
    - face_encoder_blocks: 6 blocks, max 512 channels
    - audio_encoder: no audio_refine, no SAM
    - face_decoder_blocks: starts from 512 channels
    - Input: face_sequences has 6 channels (original RGB + masked)
    - No SAM attention mechanism
    """
    def __init__(self, audio_encoder=None):
        super(Wav2LipV1, self).__init__()

        # V1 face encoder: 6 blocks, Conv2d, max 512 channels
        self.face_encoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(6, 16, kernel_size=7, stride=1, padding=3),  # 192, 192
                          Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(16, 16, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 96, 96
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 48, 48
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 24, 24
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # 12, 12
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True)),

            nn.Sequential(Conv2d(256, 512, kernel_size=3, stride=2, padding=1),  # 6, 6
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True)),
        ])

        # V1 audio encoder: 7 blocks, last layer 512 channels
        if audio_encoder is None:
            self.audio_encoder = nn.Sequential(
                Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),

                Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                Conv2d(512, 512, kernel_size=1, stride=1, padding=0))
        else:
            self.audio_encoder = audio_encoder
            for p in self.audio_encoder.parameters():
                p.requires_grad = False

        # V1 decoder: starts from 512 channels (no audio_refine, no SAM)
        self.face_decoder_blocks = nn.ModuleList([
            nn.Sequential(Conv2d(512, 512, kernel_size=1, stride=1, padding=0), ),  # + 512

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),  # 3,3
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # + 512

            nn.Sequential(Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True), ),  # 6, 6  + 256

            nn.Sequential(Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True), ),  # 12, 12  + 128

            nn.Sequential(Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True), ),  # 24, 24  + 64

            nn.Sequential(Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True), ),  # 48, 48  + 32

            nn.Sequential(Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                          Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True), ), ])  # 96, 96  + 16

        self.output_block = nn.Sequential(Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                                          nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                                          nn.Sigmoid())

    def freeze_audio_encoder(self):
        for p in self.audio_encoder.parameters():
            p.requires_grad = False

    def forward(self, audio_sequences, face_sequences):

        B = audio_sequences.size(0)

        input_dim_size = len(face_sequences.size())
        if input_dim_size > 4:
            audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
            face_sequences = torch.cat([face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0)

        audio_embedding = self.audio_encoder(audio_sequences)  # B, 512, 1, 1

        feats = []
        x = face_sequences

        for f in self.face_encoder_blocks:
            x = f(x)
            feats.append(x)

        x = audio_embedding
        for f in self.face_decoder_blocks:
            x = f(x)
            try:
                x = torch.cat((x, feats[-1]), dim=1)
            except Exception as e:
                print(x.size())
                print(feats[-1].size())
                raise e

            feats.pop()

        x = self.output_block(x)

        if input_dim_size > 4:
            x = torch.split(x, B, dim=0)  # [(B, C, H, W)]
            outputs = torch.stack(x, dim=2)  # (B, C, T, H, W)

        else:
            outputs = x

        return outputs

import argparse
import os
import random
import warnings

import pytorch_lightning
import pytorchvideo.data
import pytorchvideo.models.resnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torchmetrics
from pytorchvideo.data.clip_sampling import ClipInfo, ClipSampler
from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    ShortSideScale,
    RandomShortSideScale,
    UniformTemporalSubsample,
)
from pytorchvideo.transforms.functional import uniform_temporal_subsample
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
)
from torchvision.transforms._transforms_video import CenterCropVideo

from labeled_video_dataset import LabeledVideoDataset2

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)


def parse_args():
    parser = argparse.ArgumentParser(description='nothing here')
    parser.add_argument("--clip_len", type=int, help="clip duration in seconds", default=2)
    return parser.parse_args()


args = parse_args()


class BBoxesClipSampler(ClipSampler):

    def __init__(self, clip_duration):
        super(BBoxesClipSampler, self).__init__(clip_duration)

    def __call__(self, last_clip_time, video_duration, annotation):
        # annotation['label'] = ([start_1, end_1, label_1], [start_2, end_2, label_2], ...)
        start, end, _ = random.choice(annotation['label'])
        start, end = start.item(), end.item()
        duration = end - start

        max_possible_clip_start = min(end + duration / 2 - self._clip_duration,
                                      video_duration - self._clip_duration)
        max_possible_clip_start = max(max_possible_clip_start, 0)
        min_possible_clip_start = min(start - duration / 2,
                                      video_duration - self._clip_duration)
        min_possible_clip_start = max(min_possible_clip_start, 0)
        clip_start_sec = random.uniform(min_possible_clip_start, max_possible_clip_start)
        return ClipInfo(clip_start_sec,
                        clip_start_sec + self._clip_duration,
                        clip_index=0,
                        aug_index=0,
                        is_last_clip=True)


class SpatialPool(nn.Module):

    def __init__(self, thw=(8, 7, 7)):
        super().__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.thw = thw

    def forward(self, x):
        B, N, C = x.shape
        # [B, THW, C]
        x = x.transpose(1, 2)
        # [B, C, THW]
        x = x.reshape(B, C, *self.thw)
        # [B, C, T, H, W]
        x = self.avgpool(x).flatten(start_dim=2).transpose(1, 2)
        # [B, T, C]
        return x


def THUMOS14(
        ann_file,
        clip_sampler,
        video_sampler=torch.utils.data.RandomSampler,
        transform=None,
        video_path_prefix=""):
    # Populate keyframes list
    fps = 30
    labeled_video_paths = []
    ann = {}
    with open(ann_file, 'r') as f:
        for line in f.readlines():
            video_name, video_len, start_label, end_label, category_label = line.rstrip('\n').split(',')
            if int(video_len) < 59:
                continue
            video_frames_dir = os.path.join(video_path_prefix, video_name.rsplit('.', 1)[0])
            ann.setdefault(video_name, {}).setdefault('frame_dir', video_frames_dir)
            ann.setdefault(video_name, {}).setdefault('video_len', int(video_len) + 1)
            ann.setdefault(video_name, {}).setdefault('label', []).append(
                [int(start_label) / fps, int(end_label) / fps, int(category_label)])
    for video_ann in ann.values():
        # labels = np.zeros(int(video_ann['video_len'] + 1))
        # for box in video_ann['boxes']:
        #     labels[box[0]: box[1]] = 1
        # labels = {"label": torch.from_numpy(labels).float()}
        labeled_video_paths.append((video_ann['frame_dir'], {"label": torch.tensor(video_ann['label'])}))

    return LabeledVideoDataset2(
        labeled_video_paths=labeled_video_paths,
        clip_sampler=clip_sampler,
        transform=transform,
        video_sampler=video_sampler,
        decode_audio=False,
    )


class ClipLabels(torch.nn.Module):
    def __init__(self, clip_len=60, frames_per_clip=16, fps=30, temporal_scale=1):
        super().__init__()
        self.clip_len = clip_len
        self.frames_per_clip = frames_per_clip
        self.fps = fps
        self.t_scale = temporal_scale

    def get_binary_label(self, frame_indices, annotations):
        binary = torch.zeros_like(frame_indices)
        for ann in annotations:
            binary = binary.where((frame_indices < ann[0] * self.fps) | (frame_indices > ann[1] * self.fps),
                                  torch.tensor(1))
        binary_scaled = binary[::self.t_scale]
        for i in range(1, self.t_scale):
            binary_scaled = binary_scaled | binary[i::self.t_scale]
        return binary_scaled.float()

    def __call__(self, sample_dict):
        # clip_len = len(sample_dict["frame_indices"])
        # indices = torch.linspace(0, clip_len - 1, self.frames_per_clip)
        # self.subsample_indices = torch.clamp(indices, 0, clip_len - 1).long()
        # frame_indices_subsampled = torch.index_select(sample_dict["frame_indices"], dim=-1,
        #                                               index=self.subsample_indices)
        frame_indices_subsampled = uniform_temporal_subsample(sample_dict["frame_indices"], self.frames_per_clip,
                                                              temporal_dim=0)
        sample_dict['label'] = self.get_binary_label(frame_indices_subsampled, sample_dict["label"])
        return sample_dict


class THUMOS14DataModule(pytorch_lightning.LightningDataModule):
    # Dataset configuration
    _DATA_PATH = 'my_data/thumos14'
    _ANN_FILE_TRAIN = 'annotations/apn/apn_val.csv'
    _ANN_FILE_VAL = 'annotations/apn/apn_test_demo.csv'
    _FPS = 30
    _CLIP_DURATION = args.clie_len  # Duration of sampled clip for each video
    _FRAMES_PER_CLIP = 16
    _BATCH_SIZE = 4
    _NUM_WORKERS = 8  # Number of parallel processes fetching data

    def train_dataloader(self):
        """
        Create the Kinetics train partition from the list of video labels
        in {self._DATA_PATH}/train
        """
        train_transform = Compose(
            [
                ClipLabels(clip_len=self._CLIP_DURATION * self._FPS, frames_per_clip=self._FRAMES_PER_CLIP),
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self._FRAMES_PER_CLIP),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            RandomShortSideScale(min_size=256, max_size=320),
                            RandomCrop(224),
                            RandomHorizontalFlip(p=0.5),
                        ]
                    ),
                ),

            ]
        )
        train_dataset = THUMOS14(
            ann_file=os.path.join(self._DATA_PATH, self._ANN_FILE_TRAIN),
            clip_sampler=BBoxesClipSampler(self._CLIP_DURATION),
            video_path_prefix=os.path.join(self._DATA_PATH, "rawframes/val"),
            transform=train_transform
        )
        return torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self._BATCH_SIZE,
            num_workers=self._NUM_WORKERS,
        )

    def val_dataloader(self):
        """
        Create the Kinetics validation partition from the list of video labels
        in {self._DATA_PATH}/val
        """
        val_transform = Compose(
            [
                ClipLabels(clip_len=self._CLIP_DURATION * self._FPS, frames_per_clip=self._FRAMES_PER_CLIP),
                ApplyTransformToKey(
                    key="video",
                    transform=Compose(
                        [
                            UniformTemporalSubsample(self._FRAMES_PER_CLIP),
                            Lambda(lambda x: x / 255.0),
                            Normalize((0.45, 0.45, 0.45), (0.225, 0.225, 0.225)),
                            ShortSideScale(size=256),
                            CenterCropVideo(crop_size=224)
                        ]
                    ),
                ),
            ]
        )
        val_dataset = THUMOS14(
            ann_file=os.path.join(self._DATA_PATH, self._ANN_FILE_VAL),
            clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", self._CLIP_DURATION),
            video_path_prefix=os.path.join(self._DATA_PATH, "rawframes/test"),
            transform=val_transform
        )
        return torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self._BATCH_SIZE,
            # num_workers=self._NUM_WORKERS,
            num_workers=4,
        )


def make_kinetics_mvit():
    # pytorchvideo.models.vision_transformers.create_multiscale_vision_transformers()
    # return pytorchvideo.models.resnet.create_resnet(
    #     input_channel=3,  # RGB input from Kinetics
    #     model_depth=50,  # For the tutorial let's just use a 50 layer network
    #     model_num_class=400,  # Kinetics has 400 classes so we need out final head to align
    #     norm=nn.BatchNorm3d,
    #     activation=nn.ReLU,
    # )
    model = torch.hub.load("facebookresearch/pytorchvideo", model="mvit_base_16x4", pretrained=True)
    # for m in model.modules():
    #     if hasattr(m, 'cls_embed_on'):
    #         m.cls_embed_on = False
    #     if hasattr(m, 'has_cls_embed'):
    #         m.has_cls_embed = False
    # model.head = nn.Sequential(SpatialPool(thw=[8, 7, 7]), nn.Linear(768, 2), nn.Flatten(start_dim=1))
    model.head.proj = nn.Linear(768, 16)
    return model


class VideoClassificationLightningModule(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = make_kinetics_mvit()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        # The model expects a video tensor of shape (B, C, T, H, W), which is the
        # format provided by the dataset
        y_hat = self.model(batch["video"])

        # Compute cross entropy loss, loss.backwards will be called behind the scenes
        # by PyTorchLightning after being returned from this method.
        loss = F.binary_cross_entropy_with_logits(y_hat, batch["label"])
        accu = torchmetrics.functional.accuracy(y_hat, batch["label"].long())

        # Log the train loss to Tensorboard
        self.log("train_loss", loss.item())
        self.log("train_accu", accu, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self.model(batch["video"])
        loss = F.binary_cross_entropy_with_logits(y_hat, batch["label"])
        accu = torchmetrics.functional.accuracy(y_hat, batch["label"].long())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accu", accu, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """
        Setup the Adam optimizer. Note, that this function also can return a lr scheduler, which is
        usually useful for training video models.
        """
        return torch.optim.Adam(self.parameters(), lr=1e-5)


def train():
    classification_module = VideoClassificationLightningModule()
    data_module = THUMOS14DataModule()
    # train, val = data_module.train_dataloader().dataset, data_module.val_dataloader().dataset
    # t = next(val)
    trainer = pytorch_lightning.Trainer(accelerator="gpu", devices=1, check_val_every_n_epoch=1, max_epochs=1000)
    trainer.fit(classification_module, data_module.train_dataloader(), data_module.val_dataloader())


if __name__ == '__main__':
    train()

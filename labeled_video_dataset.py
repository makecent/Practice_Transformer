# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import torch.utils.data
from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.clip_sampling import ClipSampler
from pytorchvideo.data.frame_video import FrameVideo
from pytorchvideo.data.labeled_video_dataset import LabeledVideoDataset
from pytorchvideo.data.labeled_video_paths import LabeledVideoPaths
from pytorchvideo.data.utils import MultiProcessSampler
from pytorchvideo.data.video import VideoPathHandler

logger = logging.getLogger(__name__)


class FrameVideo2(FrameVideo):

    @classmethod
    def from_directory(
            cls,
            path: str,
            fps: float = 30.0,
            multithreaded_io=False,
            path_order_cache: Optional[Dict[str, List[str]]] = None,
            filename_tmpl=None,
    ):
        if path_order_cache is not None and path in path_order_cache:
            return cls.from_frame_paths(path_order_cache[path], fps, multithreaded_io)

        assert g_pathmgr.isdir(path), f"{path} is not a directory"
        rel_frame_paths = sorted(list(Path(path).glob(filename_tmpl)))

        frame_paths = [str(p) for p in rel_frame_paths]
        if path_order_cache is not None:
            path_order_cache[path] = frame_paths
        return cls.from_frame_paths(frame_paths, fps, multithreaded_io)


class VideoPathHandler2(VideoPathHandler):

    def video_from_path(self, filepath, decode_audio=False, decoder="pyav", fps=30, filename_tmpl='img_*.jpg'):
        try:
            is_file = g_pathmgr.isfile(filepath)
            is_dir = g_pathmgr.isdir(filepath)
        except NotImplementedError:

            # Not all PathManager handlers support is{file,dir} functions, when this is the
            # case, we default to assuming the path is a file.
            is_file = True
            is_dir = False

        if is_file:
            from pytorchvideo.data.encoded_video import EncodedVideo

            return EncodedVideo.from_path(filepath, decode_audio, decoder)
        elif is_dir:

            assert not decode_audio, "decode_audio must be False when using FrameVideo"
            return FrameVideo2.from_directory(
                filepath, fps, path_order_cache=self.path_order_cache, filename_tmpl=filename_tmpl
            )
        else:
            raise FileNotFoundError(f"{filepath} not found.")


class LabeledVideoDataset2(LabeledVideoDataset):
    def __init__(self, *args, **kwargs):
        super(LabeledVideoDataset2, self).__init__(*args, **kwargs)
        self.video_path_handler = VideoPathHandler2()

    def __next__(self) -> dict:
        if not self._video_sampler_iter:
            # Setup MultiProcessSampler here - after PyTorch DataLoader workers are spawned.
            self._video_sampler_iter = iter(MultiProcessSampler(self._video_sampler))

        for i_try in range(self._MAX_CONSECUTIVE_FAILURES):
            # Reuse previously stored video if there are still clips to be sampled from
            # the last loaded video.
            if self._loaded_video_label:
                video, info_dict, video_index = self._loaded_video_label
            else:
                video_index = next(self._video_sampler_iter)  # return StopIteration when all video used
                try:
                    video_path, info_dict = self._labeled_videos[video_index]
                    video = self.video_path_handler.video_from_path(
                        video_path,
                        decode_audio=self._decode_audio,
                        decoder=self._decoder,
                    )
                    self._loaded_video_label = (video, info_dict, video_index)
                except Exception as e:
                    logger.debug(
                        "Failed to load video with error: {}; trial {}".format(
                            e,
                            i_try,
                        )
                    )
                    continue
            (
                clip_start,
                clip_end,
                clip_index,
                aug_index,
                is_last_clip,
            ) = self._clip_sampler(
                self._next_clip_start_time, video.duration, info_dict
            )

            if isinstance(clip_start, list):  # multi-clip in each sample

                # Only load the clips once and reuse previously stored clips if there are multiple
                # views for augmentations to perform on the same clips.
                if aug_index[0] == 0:
                    self._loaded_clip = {}
                    loaded_clip_list = []
                    for i in range(len(clip_start)):
                        clip_dict = video.get_clip(clip_start[i], clip_end[i])
                        if clip_dict is None or clip_dict["video"] is None:
                            self._loaded_clip = None
                            break
                        loaded_clip_list.append(clip_dict)

                    if self._loaded_clip is not None:
                        for key in loaded_clip_list[0].keys():
                            self._loaded_clip[key] = [x[key] for x in loaded_clip_list]

            else:  # single clip case

                # Only load the clip once and reuse previously stored clip if there are multiple
                # views for augmentations to perform on the same clip.
                if aug_index == 0:
                    self._loaded_clip = video.get_clip(clip_start, clip_end)

            self._next_clip_start_time = clip_end

            video_is_null = (
                    self._loaded_clip is None or self._loaded_clip["video"] is None
            )
            if (
                    is_last_clip[-1] if isinstance(is_last_clip, list) else is_last_clip
            ) or video_is_null:
                # Close the loaded encoded video and reset the last sampled clip time ready
                # to sample a new video on the next iteration.
                self._loaded_video_label[0].close()
                self._loaded_video_label = None
                self._next_clip_start_time = 0.0
                self._clip_sampler.reset()
                if video_is_null:
                    logger.debug(
                        "Failed to load clip {}; trial {}".format(video.name, i_try)
                    )
                    continue

            frames, frame_indices = self._loaded_clip["video"], self._loaded_clip["frame_indices"]
            audio_samples = self._loaded_clip["audio"]
            sample_dict = {
                "duration": video.duration,
                "fps": video._fps,
                "video": frames,
                "frame_indices": torch.tensor(frame_indices, dtype=torch.long),
                "video_name": video.name,
                "video_index": video_index,
                "clip_index": clip_index,
                "aug_index": aug_index,
                **info_dict,
                **({"audio": audio_samples} if audio_samples is not None else {}),
            }
            if self._transform is not None:
                sample_dict = self._transform(sample_dict)

                # User can force dataset to continue by returning None in transform.
                if sample_dict is None:
                    continue

            return sample_dict
        # else:
        #     raise RuntimeError(
        #         f"Failed to load video after {self._MAX_CONSECUTIVE_FAILURES} retries."
        #     )


def labeled_video_dataset(
        data_path: str,
        clip_sampler: ClipSampler,
        video_sampler: Type[torch.utils.data.Sampler] = torch.utils.data.RandomSampler,
        transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None,
        video_path_prefix: str = "",
        decode_audio: bool = True,
        decoder: str = "pyav",
) -> LabeledVideoDataset:
    """
    A helper function to create ``LabeledVideoDataset`` object for Ucf101 and Kinetics datasets.

    Args:
        data_path (str): Path to the data. The path type defines how the data
            should be read:

            * For a file path, the file is read and each line is parsed into a
              video path and label.
            * For a directory, the directory structure defines the classes
              (i.e. each subdirectory is a class).

        clip_sampler (ClipSampler): Defines how clips should be sampled from each
                video. See the clip sampling documentation for more information.

        video_sampler (Type[torch.utils.data.Sampler]): Sampler for the internal
                video container. This defines the order videos are decoded and,
                if necessary, the distributed split.

        transform (Callable): This callable is evaluated on the clip output before
                the clip is returned. It can be used for user defined preprocessing and
                augmentations to the clips. See the ``LabeledVideoDataset`` class for clip
                output format.

        video_path_prefix (str): Path to root directory with the videos that are
                loaded in ``LabeledVideoDataset``. All the video paths before loading
                are prefixed with this path.

        decode_audio (bool): If True, also decode audio from video.

        decoder (str): Defines what type of decoder used to decode a video.

    """
    labeled_video_paths = LabeledVideoPaths.from_path(data_path)
    labeled_video_paths.path_prefix = video_path_prefix
    dataset = LabeledVideoDataset(
        labeled_video_paths,
        clip_sampler,
        video_sampler,
        transform,
        decode_audio=decode_audio,
        decoder=decoder,
    )
    return dataset

import os
import pandas as pd
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
import torchvision.transforms as Tv
import torchaudio.transforms as Ta
import torchaudio

class AV_Dataset(Dataset):
    def __init__(self, annotations_file, audio_dir, img_dir, spec_mean, spec_std, num_images_per_clip=8):
        super(AV_Dataset, self).__init__()

        self.annos = pd.read_csv(annotations_file, header=None)  # columns as [file_name, label]
        self.audio_dir = audio_dir  # all files in '.wav' format
        self.img_dir = img_dir # all '.jpg' images
        self.num_images_per_clip = num_images_per_clip

        self.visual_transforms = Tv.Compose([
            # input image (224x224) by default
            Tv.ToTensor(),
            Tv.ConvertImageDtype(torch.float32),
            # normalize to imagenet mean and std values
            Tv.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.sampling_frequency = 16000
        self.mel = Ta.MelSpectrogram(
                      sample_rate=16000,
                      n_fft=400,
                      win_length=400,
                      hop_length=160,
                      n_mels=128,
                  )
                  
        self.a2d = Ta.AmplitudeToDB()
        # mean and std already calculated.
        self.spec_mean = spec_mean
        self.spec_std = spec_std

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, idx):

        # select one clip
        clip_name = self.annos.iloc[idx, 0]

        # load the audio file with torch audio
        audio_path = os.path.join(self.audio_dir, clip_name + '.wav')
        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        # use mono audio instead os stereo audio (use left by default)
        waveform = waveform[0]

        # resample
        if sample_rate != self.sampling_frequency:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sampling_frequency)

        # normalize raw waveform
        waveform = (waveform - torch.mean(waveform)) / (torch.std(waveform) + 1e-6)
        # generate mel spectrogram and convert amplitude to decibels
        spectrogram = self.a2d(self.mel(waveform))
        # normalize spectrogram.
        spectrogram = (spectrogram - self.spec_mean) / self.spec_std
        spectrogram = spectrogram.type(torch.float32)

        # load images
        file_path = self.img_dir + clip_name + '/'
        frame_names = [i for i in os.listdir(file_path)]

        # resampling indices
        target_frame_idx = np.linspace(0, len(frame_names)-1 , num=self.num_images_per_clip, dtype=int)

        rgb_frames = []
        for i in target_frame_idx:
            img = np.asarray(Image.open(file_path + frame_names[i])) / 255.0
            rgb_frames.append(self.visual_transforms(img))

        rgb_frames = torch.stack(rgb_frames, 0).type(torch.float32)

        # assign integer to labels
        label = int(self.annos.iloc[idx, 1])
        
        return spectrogram, rgb_frames, label

# test = AV_Dataset('train.csv','audio_files','rgb_frames/')
# test_loader = DataLoader(test,batch_size=32,shuffle=True, collate_fn=collate_fn)
# for spec,frame,label in test_loader:
#     print(spec.shape, frame.shape, label.shape)
    # break    
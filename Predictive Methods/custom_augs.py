"""
Contains the following custom augmentations based on torch transforms:
    - fade in/out
    - time masking
    - time stretch

These custom augmentations are built on a version of torch-audiomentations that 
    wasn't released officially yet (cloned from github repo on 7/6/22). The main 
    differences between this version and those released is the use of ObjectDict. 
In our version here we use ObjectDict (which appears to be a new inclusion of 
    torch-audiomentations compared to official releases). This means that data 
    is released from the composed augments as a dict which has to be indexed by 
    ['samples'] to get the actual data samples back out. 
This may functionally change in the future, if it does the following files will 
    likely need edited:
        -> This file along with the custom augmentation classes
        -> aug_selection.py (may have to change how the classes stake params/
            what params it needs)
        -> dataset_utils.py (ContrastiveTransformations). In this class we 
            currently do mentioned ['samples'] indexing, this may have to be changed
Instead of changing these files etc, it would also be possible to get the exact 
    code used from torch-audiomentations from github archives. From here the code 
    can be manually added to the virtual environment.
"""
################################################################################
# IMPORTS
################################################################################
import torch 
import random
import torchaudio
from torch import Tensor
from typing import Optional
import matplotlib.pyplot as plt

from Dataset_.dataset_utils import enforce_length
from Dataset_.vocoder import phase_vocoder

from torch_audiomentations.utils.object_dict import ObjectDict
from torch_audiomentations.core.transforms_interface import BaseWaveformTransform

################################################################################
# FADE IN/OUT
################################################################################
class CustomFade(BaseWaveformTransform):
    def __init__(self, max_fade_in_ratio, max_fade_out_ratio, fade_shapes, mode='per_example', 
            sample_rate=16000, p=1.0):
        """A custom time-series fade class built upon torchaudio's Fade. Allows 
            for fade from both front and back in a variety of shapes.

        Args:
            max_fade_in_ratio (float): Maximum fade in ratio for the signal
            max_fade_out_ratio (float): Minimum fade in ratio for the signal
            fade_shapes (list): List of shapes to apply fade with
            mode (str, optional): Mode for which to apply the augmentations. 
                Defaults to 'per_example'.
            sample_rate (int, optional): Sample rate of the signal. Defaults to 16000.
            p (float, optional): Probability with which to apply the augmentation. 
                Defaults to 1.0.
        """
        super().__init__(
            sample_rate=sample_rate,
            mode=mode,
            p=p
        )

        self.fade_shapes = fade_shapes
        self.max_fade_out_ratio = max_fade_out_ratio
        self.max_fade_in_ratio = max_fade_in_ratio

    def randomise_params(self, samples):
        """Randomises params for incoming batch of augmentations

        Args:
            samples (Tensor): Incoming data batch of signals
        """
        # Get data details
        batch_size, num_channels, sample_length = samples.shape

        # Pre-calculate the appropriate maxes of fades form both sides
        fade_in_size = int(sample_length*self.max_fade_in_ratio)
        fade_out_size = int(sample_length*self.max_fade_out_ratio)

        # Create a transform for every sample in batch
        if self.mode == 'per_example':
            shapes = random.choices(self.fade_shapes, k=batch_size)

            # Generate a new transform for every sample
            self.batch_transforms = []
            for i in range(batch_size):
                in_fade =random.randint(0, fade_in_size)
                out_fade =random.randint(0, fade_out_size)

                new_trans = torchaudio.transforms.Fade(fade_in_len=in_fade,
                    fade_out_len=out_fade, fade_shape=shapes[i])
                # Store generated per example transforms
                self.batch_transforms.append(new_trans)

        # Want to use the same transform across the batch
        elif self.mode == 'per_batch':
            shape_idx = random.randint(0, len(self.fade_shapes))
            in_fade = random.randint(0, fade_in_size)
            out_fade = random.randint(0, fade_out_size)
            self.transform = torchaudio.transforms.Fade(fade_in_len=fade_in_size,
                fade_out_len=fade_out_size, fade_shape=self.fade_shapes[shape_idx])


    def apply_transform(self,
            samples: Tensor = None,
            sample_rate: Optional[int] = None,
            targets: Optional[Tensor] = None,
            target_rate: Optional[int] = None) -> ObjectDict:
        """Mandatory apply augmentation function. Is called by super class when 
            augs are composed together Actually in charge of applying the semantics 
            of the specific augmentation to the batch of samples.  

        Args:
            samples (Tensor, optional): Incoming data batch of signals. Defaults to None.
            sample_rate (Optional[int], optional): Sample rate of signal. Defaults to None.
            targets (Optional[Tensor], optional): Target data samples. Defaults to None.
            target_rate (Optional[int], optional): Target sample rate. Defaults to None.

        Returns:
            ObjectDict: Dictionary of samples and other meta data. Should be accessed as needed
        """
        batch_size, num_channels, num_samples = samples.shape

        self.randomise_params(samples)
        
        if self.mode == 'per_example':
            for i in range(batch_size):
                samples[i, ...] = self.batch_transforms[i](samples[i][None])

        elif self.mode == 'per_batch':
            for i in range(batch_size):
                samples[i, ...] = self.transform(samples[i][None])

        return ObjectDict(samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate)


################################################################################
# TIME MASKING
################################################################################
class CustomTimeMasking(BaseWaveformTransform):
    def __init__(self, max_signal_ratio, options, mode='per_example', 
            sample_rate=16000, p=1.0):
        """Custom time masking class wrapped around torchaudio TimeMasking. 
            Can either place signal with pure white noise or make constant 0. 
            Both the length of the mask and where it starts is randomly selected. 
            Location is not bound.

            (As of right nw only 0 masking works not white noise!)

        Args:
            max_signal_ratio (float): Maximum ratio of the signal that can be masked at any given time
            options (list): List of application options. White Noise or 0 padding 
            mode (str, optional): Mode for which to apply the augmentations. 
                Defaults to 'per_example'.
            sample_rate (int, optional): Sample rate of the signal. Defaults to 16000.
            p (float, optional): Probability with which to apply the augmentation. 
                Defaults to 1.0.
        """
        super().__init__(
            sample_rate=sample_rate,
            mode=mode,
            p=p
        )

        self.max_signal_ratio = max_signal_ratio
        self.options = options 

    def randomise_params(self, samples):
        """Randomises params for incoming batch of augmentations

        Args:
            samples (Tensor): Incoming data batch of signals
        """
        # should add in noise masking later as well

        # Get data details
        batch_size, num_channels, sample_length = samples.shape

        # Pre-calculate the appropriate max mask
        max_mask = self.max_signal_ratio * sample_length

        # Generates a random use constant value masking, every time called new mask applied
        self.constant_aug = torchaudio.transforms.TimeMasking(
                time_mask_param=max_mask)

    
    def apply_transform(self,
            samples: Tensor = None,
            sample_rate: Optional[int] = None,
            targets: Optional[Tensor] = None,
            target_rate: Optional[int] = None) -> ObjectDict:
        """Mandatory apply augmentation function. Is called by super class when 
            augs are composed together Actually in charge of applying the semantics 
            of the specific augmentation to the batch of samples.  

        Args:
            samples (Tensor, optional): Incoming data batch of signals. Defaults to None.
            sample_rate (Optional[int], optional): Sample rate of signal. Defaults to None.
            targets (Optional[Tensor], optional): Target data samples. Defaults to None.
            target_rate (Optional[int], optional): Target sample rate. Defaults to None.

        Returns:
            ObjectDict: Dictionary of samples and other meta data. Should be accessed as needed
        """
        batch_size, num_channels, num_samples = samples.shape

        self.randomise_params(samples)
        
        if self.mode == 'per_example':
            for i in range(batch_size):
                samples[i, ...] = self.constant_aug(samples[i][None])

        return ObjectDict(samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate)


################################################################################
# TIME STRETCH
################################################################################
class CustomTimeStretch(BaseWaveformTransform):
    def __init__(self, min_stretch, max_stretch, n_fft=2048, mode='per_example', 
            sample_rate=16000, p=1.0):
        super().__init__(
            sample_rate=sample_rate,
            mode=mode,
            p=p
        )

        # Pre-calculated the necessary stft variables
        self.n_fft = n_fft
        # This calculation is a rearranged form of an equation in torch stft source (init function)
        self.n_freq = int((self.n_fft / 2) + 1)

        self.min_stretch = min_stretch
        self.max_stretch = max_stretch

    
    def randomise_params(self, samples):
        # Get data details
        batch_size, num_channels, sample_length = samples.shape

        # Gets a batch size worth of valid stretching parameters
        self.rand_nums = []
        for _ in range(batch_size):
            rand_num = random.uniform(self.min_stretch, self.max_stretch)
            self.rand_nums.append(round(rand_num, 2))


    def apply_transform(self,
            samples: Tensor = None,
            sample_rate: Optional[int] = None,
            targets: Optional[Tensor] = None,
            target_rate: Optional[int] = None) -> ObjectDict:
        """Mandatory apply augmentation function. Is called by super class when 
            augs are composed together Actually in charge of applying the semantics 
            of the specific augmentation to the batch of samples.  

        Args:
            samples (Tensor, optional): Incoming data batch of signals. Defaults to None.
            sample_rate (Optional[int], optional): Sample rate of signal. Defaults to None.
            targets (Optional[Tensor], optional): Target data samples. Defaults to None.
            target_rate (Optional[int], optional): Target sample rate. Defaults to None.

        Returns:
            ObjectDict: Dictionary of samples and other meta data. Should be accessed as needed
        """

        batch_size, num_channels, num_samples = samples.shape
        # Make a call to randomise params or batch of augs
        self.randomise_params(samples)

        if self.mode == 'per_example':
            for i in range(batch_size):
                # Generates real and imaginary stft data
                stft_data = torch.stft(samples[i], n_fft=2048)
                # Stretches the stft data using a phase vocoder
                stretched_stft = phase_vocoder(stft_data, self.rand_nums[i],phase_advance=self.n_freq )
                # Recombines the real and imaginary component of stft with an inverse 
                recombined_data = torch.istft(stretched_stft, n_fft=self.n_fft)
                # Uses circular padding or cropping to return to expected signal length
                samples[i, ...] = enforce_length(recombined_data, num_samples)


        return ObjectDict(samples=samples,
            sample_rate=sample_rate,
            targets=targets,
            target_rate=target_rate)





################################################################################
# TESTING
################################################################################
# import time
# from Dataset_.dataset_utils import ContrastiveTransformations
# from torch_audiomentations import Compose, Gain, PolarityInversion, AddBackgroundNoise, BandPassFilter, BandStopFilter, \
#     AddColoredNoise, HighPassFilter, ApplyImpulseResponse, LowPassFilter,  PeakNormalization, PitchShift, \
#          Shift, ShuffleChannels, TimeInversion

# custom_trans = CustomFade(10, 10, ['linear'])
# apply = Compose(transforms=[custom_trans])

# exaple_data = torch.rand(1000, 1, 160000).to('cuda')

# contrast_augs_list = Compose(transforms=[PitchShift(sample_rate=16000, 
#         min_transpose_semitones=-15, 
#         max_transpose_semitones=15,
#         p=0.0)
# ,
#     CustomFade(0.5, 0.5, ['linear', 'logarithmic', 'exponential'], p=0, sample_rate=16000),
#     CustomTimeMasking(0.125, ['constant'], p=0, sample_rate=16000),
#     CustomTimeStretch(0.5, 1.5, p=1.0)
    
#     ], shuffle=False)

# apply_cont_augs = ContrastiveTransformations(contrast_augs_list, 2)

# start = time.time()
# new_data = apply_cont_augs(exaple_data)
# end = time.time()
# print('yo:', end-start)

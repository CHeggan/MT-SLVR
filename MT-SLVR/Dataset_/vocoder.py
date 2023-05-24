import torch
import math 

# Code from https://github.com/keunwoochoi/torchaudio-contrib/blob/master/torchaudio_contrib/functional.py

def angle(complex_tensor):
    """
    Return angle of a complex tensor with shape (*, 2).
    """
    return torch.atan2(complex_tensor[..., 1], complex_tensor[..., 0])

def phase_vocoder(complex_specgrams, rate, phase_advance):
    """
    Phase vocoder. Given a STFT tensor, speed up in time
    without modifying pitch by a factor of `rate`.
    Args:
        complex_specgrams (Tensor):
            (*, channel, num_freqs, time, complex=2)
        rate (float): Speed-up factor.
        phase_advance (Tensor): Expected phase advance in
            each bin. (num_freqs, 1).
    Returns:
        complex_specgrams_stretch (Tensor):
            (*, channel, num_freqs, ceil(time/rate), complex=2).
    Example:
        >>> num_freqs, hop_length = 1025, 512
        >>> # (batch, channel, num_freqs, time, complex=2)
        >>> complex_specgrams = torch.randn(16, 1, num_freqs, 300, 2)
        >>> rate = 1.3 # Slow down by 30%
        >>> phase_advance = torch.linspace(
        >>>    0, math.pi * hop_length, num_freqs)[..., None]
        >>> x = phase_vocoder(complex_specgrams, rate, phase_advance)
        >>> x.shape # with 231 == ceil(300 / 1.3)
        torch.Size([16, 1, 1025, 231, 2])
    """
    ndim = complex_specgrams.dim()
    time_slice = [slice(None)] * (ndim - 2)

    time_steps = torch.arange(0, complex_specgrams.size(
        -2), rate, device=complex_specgrams.device)

    alphas = torch.remainder(time_steps,
                             torch.tensor(1., device=complex_specgrams.device))
    phase_0 = angle(complex_specgrams[time_slice + [slice(1)]])

    # Time Padding
    complex_specgrams = torch.nn.functional.pad(
        complex_specgrams, [0, 0, 0, 2])

    complex_specgrams_0 = complex_specgrams[time_slice +
                                            [time_steps.long()]]
    # (new_bins, num_freqs, 2)
    complex_specgrams_1 = complex_specgrams[time_slice +
                                            [(time_steps + 1).long()]]

    angle_0 = angle(complex_specgrams_0)
    angle_1 = angle(complex_specgrams_1)

    norm_0 = torch.norm(complex_specgrams_0, dim=-1)
    norm_1 = torch.norm(complex_specgrams_1, dim=-1)

    phase = angle_1 - angle_0 - phase_advance
    phase = phase - 2 * math.pi * torch.round(phase / (2 * math.pi))

    # Compute Phase Accum
    phase = phase + phase_advance
    phase = torch.cat([phase_0, phase[time_slice + [slice(-1)]]], dim=-1)
    phase_acc = torch.cumsum(phase, -1)

    mag = alphas * norm_1 + (1 - alphas) * norm_0

    real_stretch = mag * torch.cos(phase_acc)
    imag_stretch = mag * torch.sin(phase_acc)

    complex_specgrams_stretch = torch.stack(
        [real_stretch, imag_stretch],
        dim=-1)

    return complex_specgrams_stretch
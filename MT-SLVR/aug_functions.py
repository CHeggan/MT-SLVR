"""
Script deals with generating the selected augmentation fruitions that can then
    be passed to compose. Some things to note:
        - Ordering of the augmentations matters
        - If selected the probability should always be set to 1, we are applying
            these augs in order to create correlated views. This is not a normal 
            augmentation scheme
        - We make heavy use of the torch-audiomentations package but do use 
            some of our own functions wrapped around torch audio transforms and 
            functionals
"""
################################################################################
# IMPORTS
################################################################################
import torch
import random
from Dataset_.dataset_utils import ContrastiveTransformations
from custom_augs import CustomFade, CustomTimeMasking, CustomTimeStretch
from torch_audiomentations import Compose, PitchShift, AddColoredNoise, Shift

################################################################################
# AUG SELECTION FUNCTION
################################################################################
def select_possible_augs(augs_to_exclude, aug_params, sample_rate=16000):
    """Function to select and initialise augmentation classes based on 
        keyword and pre-defined parameter dictionary

    Args:
        augs_to_exclude (list): List of augmentations to not be included, these are the 
            keywords as input into config file
        aug_params (dict): Dictionary of augmentation parameters
        sample_rate (int, optional): Sample rate of teh signal being used. 
            Defaults to 16000.

    Raises:
        ValueError: If augmentation keyword is not recognised, we throw an error

    Returns:
        list: List of initialised augmentation classes
    """
    all_possible_augs = ['pitch_shift', 'fade', 'white_noise', 'mixed_noise', 
        'time_stretch', 'time_masking', 'time_shift']

    for aug in augs_to_exclude:
        all_possible_augs.remove(aug)

    augs_to_include = all_possible_augs

    p = aug_params['p']
    aug_list = []

    for name in augs_to_include:
        if name == 'pitch_shift':
            aug = PitchShift(**aug_params[name], p=p, sample_rate=sample_rate)

        elif name == 'fade':
            aug = CustomFade(**aug_params[name], p=p, sample_rate=sample_rate)

        elif name == 'white_noise':
            aug = AddColoredNoise(**aug_params[name], p=p, sample_rate=sample_rate)

        elif name == 'mixed_noise':
            aug = AddColoredNoise(**aug_params[name], p=p, sample_rate=sample_rate)

        elif name == 'time_masking':
            aug = CustomTimeMasking(**aug_params[name], p=p, sample_rate=sample_rate)

        elif name == 'time_shift':
            aug = Shift(**aug_params[name], p=p, sample_rate=sample_rate)

        elif name == 'time_stretch':
            aug = CustomTimeStretch(**aug_params[name], p=p, sample_rate=sample_rate)

        elif name == 'None':
            continue

        else:
            raise ValueError(f'Augmentation name: {name} not recognised')

        aug_list.append(aug)

    return aug_list


################################################################################
# GENERATE AUGMENTATION PIPELINES
################################################################################
def gen_multi_label_augs(possible_augs, params, batch_size, device, batch_wise):
    """Augmentation pipeline and multi-label y array generation function. 
        Generation is done using some constraints like number of augs to 
        apply at any given time. Function allows for both the creation of a 
        single aug prescription as well as a list of length 'batch_size' prescriptions. 
        This is to facilitate the scenarios (a) a batch has the same selection and 
        ordering of augs applied across all contained samples to create transformed 
        views, and (b)each sample instance is treated separately. 

    Args:
        possible_augs (list of functions): All possible augmentation functions 
            being used in this training pipeline
        params (dict): Experimental params
        batch_size (int): Number of samples in a given batch. Only used if doing
            sample-wise transform creation
        device (torch object): Device to map data to
        batch_wise (bool, optional): Whether or not to create a single transform 
            pipeline for a whole batch. Defaults to True.

    Returns:
        list or list of lists & torch tensor: Transformation pipelines for contrastive learning and batch label tensor
    """

    if batch_wise:
        # Create an initial index list for multi-label tensors
        y = torch.zeros(size=(len(possible_augs), )).to(device)

        if params['aug_control']['fixed_k'] == True:
            k = params['aug_control']['k']

        elif params['aug_control']['fixed_k'] == False:
            max_k = len(possible_augs)
            k = random.randint(1, max_k)

        aug_idxs = random.sample(list(range(len(possible_augs))), k=k)

        y[aug_idxs] = 1
        rand_aug_list = [possible_augs[i] for i in aug_idxs]
        augs = Compose(rand_aug_list, shuffle=False)

        y = y.repeat(batch_size, 1)


    elif not batch_wise:
        # Store all augmentation pipelines
        augs = []
        max_k = len(possible_augs)
        # Initialise a y label array 
        y = torch.zeros(size=(batch_size, len(possible_augs))).to(device)

        for idx in range(batch_size):

            if params['aug_control']['fixed_k'] == True:
                k = params['aug_control']['k']

            else:
                k = random.randint(1, max_k)

            aug_idxs = random.sample(list(range(len(possible_augs))), k=k)

            y[idx][aug_idxs] = 1
            rand_aug_list = [possible_augs[i] for i in aug_idxs]
            comp_augs = Compose(rand_aug_list, shuffle=False)
            augs.append(comp_augs)

    return augs, y

################################################################################
# GENERATE AUGMENTATION PIPELINES (NOT USED HERE)
################################################################################
def gen_aug_funcs(possible_augs, params, batch_size, batch_wise=True):
    """Augmentation pipeline generation function. Generation is done using some 
        constraints like number of augs to apply at any given time. Function 
        allows for both the creation of a single aug prescription as well as a
        list of length 'batch_size' prescriptions. This is to facilitate the
        scenarios (a) a batch has the same selection and ordering of augs 
        applied across all contained samples to create transformed views, and (b)
        each sample instance is treated separately. 

    Args:
        possible_augs (list of functions): All possible augmentation functions 
            being used in this training pipeline
        params (dict): Experimental params
        batch_size (int): Number of samples in a given batch. Only used if doing
            sample-wise transform creation
        batch_wise (bool, optional): Whether or not to create a single transform 
            pipeline for a whole batch. Defaults to True.

    Returns:
        list or list of lists: Transformation pipelines for contrastive learning
    """

    if batch_wise:

        if params['aug_control']['fixed_k'] == True:
            k = params['aug_control']['k']

        elif params['aug_control']['fixed_k'] == False:
            max_k = len(possible_augs)
            k = random.randint(1, max_k)

        rand_augs = random.choices(possible_augs, k=k)
        comp_augs = Compose(rand_augs, shuffle=False)
        apply_cont_augs = ContrastiveTransformations(comp_augs, n_views=2)

        return apply_cont_augs


    elif not batch_wise:
        # Store all augmentation pipelines
        sample_wise_augs = []
        max_k = len(possible_augs)

        # Generate batch size number of combinations
        for _ in range(batch_size):

            if params['aug_control']['fixed_k'] == True:
                k = params['aug_control']['k']

            else:
                k = random.randint(1, max_k)

            rand_augs = random.choices(possible_augs, k=k)
            comp_augs = Compose(rand_augs, shuffle=False)
            apply_cont_augs = ContrastiveTransformations(comp_augs, n_views=2)
            sample_wise_augs.append(apply_cont_augs)

        return sample_wise_augs

################################################################################
# APPLY AUGMENTATION PIPELINES TO A BATCH
################################################################################
def apply_aug_funcs(samples, augs_to_apply, batch_wise):
    """Application of generate aug pipelines function. Can either deal with the 
        trivial batch-wise case or the more complex sample-wise case (every instance 
        has its own randomly generated transform pipeline). Function is highly 
        related/dependant on the function 'gen_aug_funcs' and is only split out 
        to make things more clear

    Args:
        samples (torch tensor): The original batched data samples
        augs_to_apply (class or list of classes): Instances(s) of the contrastive 
            transformation class which contain pipelines for data augmentation
        batch_wise (bool): Whether to apply transforms per batch or per sample

    Returns:
        list of torch tensors: Final list of augmented tensors
    """
    # Batch -wise is an easy variant of this as we pass the whole batch through 
    #   the same transformation setup
    if batch_wise:
        augs_to_apply = ContrastiveTransformations(augs_to_apply, n_views=2)
        samples = augs_to_apply(samples)
        # [x1,x2]
        return samples

    # Sample-wise has to iterate over the full batch and recombine tensors 
    else:
        final_samples_1, final_samples_2 = [], []

        for idx in range(samples.shape[0]):
            sub_aug_to_apply = augs_to_apply[idx]
            sub_aug_to_apply = ContrastiveTransformations(sub_aug_to_apply, n_views=2)
            # Need to unsqueeze as slice collapses batch dimension
            trans_sample_idx = sub_aug_to_apply(samples[idx].unsqueeze(0))

            # Splits out and stores the two batches into separate lists for concat
            final_samples_1.append(trans_sample_idx[0])
            final_samples_2.append(trans_sample_idx[1])

        # Final batch concatenation into out two separate batched sample views
        final_samples_1 = torch.cat(final_samples_1, dim=0)
        final_samples_2 = torch.cat(final_samples_2, dim=0)

        # [x1,x2]
        return [final_samples_1, final_samples_2]
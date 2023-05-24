from aug_functions import select_augs
from Dataset_.dataset_utils import ContrastiveTransformations
from torch_audiomentations import Compose

import time
import yaml

with open("Dataset_/aug_params.yaml") as stream:
    aug_params = yaml.safe_load(stream)

augs_to_use = ['pitch_shift', 'fade', 'mixed_noise']
possible_augs = select_augs(augs_to_use, aug_params)
params = {'aug_control': 
    {'fixed_k': True, 'k': 2, 'batch_wise': True}
    }


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

        rand_augs = random.sample(possible_augs, k=k)
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

            rand_augs = random.sample(possible_augs, k=k)
            comp_augs = Compose(rand_augs, shuffle=False)
            apply_cont_augs = ContrastiveTransformations(comp_augs, n_views=2)
            sample_wise_augs.append(apply_cont_augs)

        return sample_wise_augs


all_times = []
for i in range(100):
    start = time.time()

    apply_aug_funcs = gen_aug_funcs(possible_augs, params, batch_size=512, batch_wise=params['aug_control']['batch_wise'])
    end = time.time()
    all_times.append(end-start)

print(sum(all_times)/len(all_times))
# MT-SLVR: Multi-Task Self-Supervised Learning for Transformation In(Variant) Representations
A novel multi-task self-supervised learning approach, capable of learning both augmentation invariant and equivariant features in a parameter efficient manner.  


<img src=images/logo.png data-canonical-src=images/logo.png width="100" height="100" />

## News & Citation
 - 1/6/23: Blog post with additional details and diagrams released: https://cheggan.github.io/posts/2023/05/MT-SLVR_blog/
 - 29/5/23: Paper and code made public
 - 17/5/23: MT-SLVR accepted to InterSpeech23, to be presented in August 


## Citation
Citation to be added

# MT-SLVR

**Schematic of MT-SLVR to be added**

Simply put, the MT-SLVR algorithm utilises multi-task learning between contrastive and predictive self-supervised learning techniques. These features learnt by each of these algorithm are expected to be heavily conflicting (i.e one tries to learn augmentation invariance while the other tries to learn augmentation equivariance). To allow both to co-exist and be readily available for downstream tasks, we utilise adapters fit throughout the neural network, allowing each task (contrastive/predictive) some of its own specific parameters to learn upon. 



# Using the Repo(s)
## Contents
This repo contains a few distinct parts which can be used to both reproduce the results from our work and train new models for varying purposes. Within this repo we include the following sub-codebases:
 - [Contrastive Only Methods](https://github.com/CHeggan/MT-SLVR/tree/main/Contrastive%20Methods): Our baseline SimSiam and SimCLR only approaches
 - [Predictive Methods](https://github.com/CHeggan/MT-SLVR/tree/main/Predictive%20Methods): Our baseline transformation prediction only approach
 - [Multi-Task Method (MT-SLVR)](https://github.com/CHeggan/MT-SLVR/tree/main/MT-SLVR): Our novel multi-task approach

We note that although there are unique parts to each of three major codebases, there is a significant amount of overlapping code, e.g. dataset and augmentation classes. We left the overall codebase like this instead of reformatting and removing repeated scripts so that each section can be used independently, effectively increasing the immediate usability of the repo. 

## Evaluation Framework
We exclude our evaluation framework from this specific repo (due to its additional complexity and potential usefulness as a standalone codebase) and instead host it [here](https://github.com/CHeggan/Few-Shot-Classification-for-Audio-Evaluation). **This evaluation repo is still under construction with respect to documentation**. 

## Environment
We use miniconda for our experimental setup. For the purposes of reproduction we include the environment file. This can be set up using the following command
```
conda env create --file torch_gpu_env.txt
```
There are likely some redundant packages in this section, we will attempt to trim it down for future releases. 

## Datasets & Processing
 For pre-training we use the balanced version of [AudioSet](https://research.google.com/audioset/). The decision to use this set was based in ease and manageable size. Unfortunately this set is not easily available to download. This being said, the set can be reproduced using a YouTube scraping script. Details and references for this process can be found [here](https://github.com/CHeggan/AudioSet-For-Meta-Learning).


## MT-SLVR Pre-Training
Additional details on how to run the the MT-SLVR can be found in its sub-codebase but the main line is of the format:

```
python NEW_RUN.py --cont_framework simclrv1 --pred_framework trans --pred_weight 1.0 --adapter parallel --num_splits 2 --batch_size 100 --lr 0.00005 --p 1.0 --data_name AS_BAL --dims 2 --in_channels 3 --model_fc_out 1000 --gpu 0
```
Hyperparameter descriptions can be found in the ["NEW_RUN.py"](https://github.com/CHeggan/MT-SLVR/blob/main/MT-SLVR/NEW_RUN.py).

## Baseline Codebases
Details on running baselines can be found in their respective sub-codebases. 

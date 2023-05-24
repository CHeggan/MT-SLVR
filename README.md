# MT-SLVR: Multi-Task Self-Supervised Learning for Transformation In(Variant) Representations
A novel multi-task self-supervised learing pproach, capable of learning both augmenattion invarinat and equivariant features in a parameter effecient manner.  

## News & Citation
 - Blog and more informal breakdown coming soon! 
 - 29/5/23: Paper and code made public
 - 17/5/23: MT-SLVR accepted to InterSpeech23, to be presented in August 



## Citation
Citation to be added

# MT-SLVR
## General Framework

add in schematic of MT-SLVR

Simply put, the MT-SLVR algorithm utilises multi-task learning between contrastive and predictive self-supervised learning techniques. These features learnt by each of these algorithm are expected to be heavily conflicting (i.e one tries to learn augmentation invariance while the other tries to learn augmentation equivariance). To allow both to co-exist and be readily available for downstream tasks, we utilise adapters fit throughout the neural network, allowing each task (contrastive/predictive) some of its own specific parameters to learn upon. 

# Using the Repo(s)

## Contents
This repo contains a few distinct parts which can be used to both reproduce the results from our work and train new models for varying purposes. Within this repo we include the following sub-codebases:
 - [Contrastive Only Methods](): Our baseline SimSiam and SimCLR only approaches
 - [Predictive Methods](): Our baseline transformation prediction only approach
 - [Multi-Task Method (MT-SLVR)](): Our novel multi-task approach

We note that although there are unique parts to each of three major codebases, there is a significant amount of overlapping code, e.g. dataset and augmentation classes. We left the overall codebase like this instead of reformatting and removing repeated scripts so that each section can be used independently, effectively increasing the immediate usability of the repo. 

## Evaluation Framework
We exclude our evaluation framework from this specific repo (due to its additional complexity and potential usefulness as a standalone codebase) and instead host it [here](). 

## Enviroment
We use miniconda for our experimental setup. For the purposes of reproduction we include the environment file. This can be set up using the following command
```
conda env create --file torch_gpu_env.txt
```
There are likely some redundant packages in this section, we will attempt to trim it down for future releases. 

## Datasets & Processing

## MT-SLVR Pre-Training

## Contrastive & Predictive Baselines









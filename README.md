# GFI_AP-Code
Heading: Adaptive CNN filter pruning using global importance metric

Description of functions (for GFI_AP_CIFAR & GFI_AP_IMAGENET):
1)	variable_list: assigns variables like dataset and model details and path for restoring pretrained models and storing results 
2)	API: contains all functions to modify the structure of the VGG models (backbone)
3)	API_multi: same as API, only difference is that it is ‘multi-gpu’ version of API, contains all functions to modify the structure of the ResNet models (backbone)
4)	pruning_API: contains functions related to pruning like finding normalized mean & importance score, pruning a model, retraining & finetuning
5)	prune_main: main script to perform pruning, retraining and finetuning
6)	flops_API: contains functions to compute FLOPs for a CNN
7)	flops_main: main script to compute FLOPs for base model and pruned model
8)	pruning_plots_main: main script to generate the plots to observe the effects of pruning

Models support: 
Given code supports pruning of filters from these models which are listed below-
VGG (11, 13, 16, 19)
ResNet (18, 34, 50, 101, 152, 20, 32, 44, 56, 110)

Results: 
Some results from our paper-

Sample checkpoints:
1)	pretrained_checkpoints:

2)	checkpoints_after_pruning:


Citation:

Acknowledgement:




# scAgent: A Versatile Single-cell Multi-omics Data Analysis Framework via Multi-agent Collaboration

## Introduction
scAgent is a versatile framework for single-cell multi-omics data analysis, which injects LLMs' priors via multi-agent collaboration.

![image]([imgs/framework.png](https://github.com/zfkarl/scAgent/blob/master/imgs/scagent-framework.png))

## Getting Started
#### Requirements
- Python 3.10, PyTorch>=1.21.0,  numpy>=1.24.0, are required for the current codebase.

#### LLM Embeddings
##### 1. Cell-level Text Embeddings
We use the Vicuna-7B model to extract the cell-level text embeddings. Download embeddings from https://drive.google.com/drive/folders/1aArcZjDckc7my9gPvVqN0h8X-7a0brLV.

##### 2. Feature-level Text Embeddings 
The original embeddings can be downloaded from https://sites.google.com/yale.edu/scelmolib. We also provide the preprocessed version in https://drive.google.com/drive/folders/1aArcZjDckc7my9gPvVqN0h8X-7a0brLV.

#### Datasets
##### CITE-seq and ASAP-seq Data 
Download dataset from https://github.com/SydneyBioX/scJoint/blob/main/data.zip.

#### Cell Type Annotation 
##### Pre-training on CITE-seq Data 
<pre>sh pretrain_cite.sh </pre> 

##### Fine-tuning on CITE-seq Data 
<pre>sh finetune_cite.sh </pre> 

##### Pre-training on ASAP-seq Data 
<pre>sh pretrain_asap.sh </pre> 

##### Fine-tuning on ASAP-seq Data 
<pre>sh finetune_asap.sh </pre> 

## Acknowledgement
Our codebase is built based on scCLIP, timm, transformers, and Pytorch Lightning. We thank the authors for the nicely organized code!

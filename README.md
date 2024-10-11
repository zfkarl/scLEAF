# scLEAF: Large Language Models Enhance Single-cell Multi-omics Biology

## Introduction
scLEAF is a versatile framework for single-cell multi-omics data analysis, which transfers cell representations to the LLM text space.

![image](https://github.com/zfkarl/scLEAF/blob/master/imgs/framework.png)

## Getting Started
#### Requirements
- Python 3.10, PyTorch>=1.21.0,  numpy>=1.24.0, are required for the current codebase.

#### Datasets
**CITE-seq and ASAP-seq Data:** download dataset from https://github.com/SydneyBioX/scJoint/blob/main/data.zip.

#### Cell Type Annotation 
##### Pre-training on CITE-seq Data 
<pre>sh pretrain_cite.sh </pre> 

##### Fine-tuning on CITE-seq Data 
<pre>sh finetune_cite.sh </pre> 

##### Pre-training on ASAP-seq Data 
<pre>sh pretrain_asap.sh </pre> 

##### Fine-tuning on ASAP-seq Data 
<pre>sh finetune_asap.sh </pre> 

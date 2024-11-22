<p align="center">

  <h1 align="center">SyncViolinist: Music-Oriented Violin Motion Generation Based on Bowing and Fingering</h1>
  <p align="center">
    <a href="https://"><strong>Hiroki Nishizawa</a></strong>*
    ·
    <a href="https://sites.google.com/view/keitarotanaka/"><strong>Keitaro Tanaka</a></strong>*
    ·
    <a href="http://"><strong>Asuka Hirata</a></strong>*
    ·
    <a href="https://sites.google.com/site/yamaguchishugo/"><strong>Shugo Yamaguchi</strong></a><br>
    <a href="http://qfeng.me"><strong>Qi Feng</strong></a>
    ·
    <a href="https://gttm.jp/hamanaka/"><strong>Masatoshi Hamanaka</strong></a>
    ·
    <a href="https://morishima-lab.jp/members?lang=en"><strong>Shigeo Morishima</strong></a><br>
    (* - Equal contribution)
  </p> 
  <!-- Upload the teaser image to the /images for it show.Update authors' homepages if necessary. -->

  <a href="">
    <img src="./images/teaser.png" alt="Teaser" width="100%"> 
    <!-- Upload the teaser image to the /images for it show. -->
  </a>


<p align="center">
  <br>
    <a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
    <a href='https://arxiv.org/abs/'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='Paper PDF'>
    </a>
    <!-- Update the link to Arxiv after submission. -->
    <a href='https://github.com/Kakanat/SyncViolinist' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'><br></br>

[SyncViolinist](https://github.com/Kakanat/SyncViolinist) is a multi-stage end-to-end framework that generates synchronized violin performance motion solely from audio input. For more details please refer to the [Paper](https://arxiv.org/abs/).

For more details check out the YouTube video below.
[![Video](images/video_teaser_play.png)](https://www.youtube.com/watch?v=)
<!-- Upload the youtube link and the thumbnail image to the /images for it show. -->


## Table of Contents

  * [Description](#description)
  * [Requirements](#requirements)
  * [Installation](#installation)
  * [Getting Started](#getting-started)
  * [Examples](#examples)
  * [Citation](#citation)
  * [Acknowledgments](#acknowledgments)
  * [Contact](#contact)



## Description

This repository includes the code base for the SyncViolinst and captured dataset.

[](# Please populate the following section after cleaning up the codebase. Feng -> Nishizawa).

## Requirements

This package has been tested for the following:

* [Pytorch>=1.7.1](https://pytorch.org/get-started/locally/) 
* Python >=3.7.0

## Installation

To install the dependencies please follow the next steps:

- Clone this repository: 
    ```Shell
    git clone https://github.com/Kakanat/SyncViolinist.git
    cd SyncViolinist
    ```
- Install the dependencies by the following commands:
    ```
    pip install -r requirements.txt
    ```

## Getting started

In order to run SyncViolinst, download the dataset and create a `data/` directory and follow the steps below:

#### SyncViolinst Dataset

- Fill in the consent form from [here](https://).
- Download the main dataset from [here](https://). After decompressing, it will include ...
- Store files under `SyncViolinst/data/`.
- The final structure of data should look as below:
```bash
    SyncViolinst
    ├── data
    │   ├── xxx
    │   │      
    .
    .
```

#### Pre-trained Checkpoints
- Download the Pre-trained checkpoints (`last.ckpt`) from [here](https://).
- Place the pre-trained models in `SyncViolinst/models` as follows:
```bash
    models
    ├── xxx.ckpt
    └── yyy.pth
```


## Examples

After installing the dependencies and downloading the data and the models, you should be able to run the following examples:
                                            
    ```Shell
    python run.py 
    ```
    The result will be saved in `SyncViolinst/save`.


## Citation

```
@inproceedings{}
```


## Acknowledgments

This research is supported by JSPS KAKENHI No. 21H05054, 24H00742, and 24H00748.

## Contact

This repository is maintained by [Hiroki Nishizawa](https://), [Keitaro Tanaka](https://sites.google.com/view/keitarotanaka/) and [Qi Feng](https://qfeng.me).

For questions, please contact [phys.keitaro1227@ruri.waseda.jp](mailto:phys.keitaro1227@ruri.waseda.jp).

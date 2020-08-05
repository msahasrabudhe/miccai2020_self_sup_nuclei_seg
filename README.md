# Self-supervised nuclei segmentation

Code to train a self-supervised segmentation network for segmentation of nuclei in histopathology images [1]. 

* ```train.py``` contains training code and defines command line options. 
* ```datasets.py``` defines datasets used to read images. 
* ```models.py``` defines relevant models (attention network and scale network).
* ```utils.py``` defines extra useful functions.
* ```configs/``` defines ```.yaml``` configuration files to set experiment parameters. 

## Installation
The Anaconda environment is specified in ```conda_env.yml```. The environment can be recreated using

```
conda env create -f conda_env.yml
```

Tested with Nvidia GeForce GTX 1080 and GeForce GTX 1080 Ti GPUs, running driver version 410.48 and cuda 10.0, and Pytorch 1.1.0 with torchvision 0.3.0.

## Data
Please see the directory [data processing](data_processing/README.md) for instructions on downloading and using data. 

## Usage
```train.py``` is the training code which offers three command line parameters. 
* ```--cfg``` specifies the configuration file to use.
* ```--gpu``` specifies which GPU to use. A value of ```-1``` implies no GPU.
* ```--output_dir``` specifies directory to record results. If the configuration file is ```name.yaml```, results will be recorded in ```<output_dir>/name```. 

Example train usage---

```
python train.py --cfg configs/example.yaml --gpu 0 --output_dir /path/to/output/
```

Example testing usage---
```
python test.py --cfg configs/example.yaml --epoch 200 --dataroot /path/to/test/imgs/ --ext tif --gpu 0 --output_dir /path/to/output/
```


## Config files
See [configuration options](configs/README.md) for a description of configuration options


## Reference
*[1] Self-Supervised Nuclei Segmentation in Histopathological Images Using Attention*, Mihir Sahasrabudhe, Stergios Christodoulidis, Roberto Salgado, Stefan Michiels, Sherene Loi, Fabrice Andre, Nikos Paragios, Maria Vakalopoulou, MICCAI 2020 [ [PDF](https://arxiv.org/pdf/2007.08373.pdf) ]

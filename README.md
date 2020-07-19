# Self-supervised nuclei segmentation (MICCAI 2020)

Code to train a self-supervised segmentation network for segmentation of nuclei in histopathology images. 

* ```train.py``` contains training code and defines command line options. 
* ```datasets.py``` defines datasets used to read images. 
* ```models.py``` defines relevant models (attention network and scale network).
* ```utils.py``` defines extra useful functions.
* ```configs/``` defines ```.yaml``` configuration files to set experiment parameters. 

## Usage
```train.py``` is the training code which offers three command line parameters. 
* ```--cfg``` specifies the configuration file to use.
* ```--gpu``` specifies which GPU to use. A value of ```-1``` implies no GPU.
* ```--output_dir``` specifies directory to record results. If the configuration file is ```name.yaml```, results will be recorded in ```output/name```. 


## Config files
See [configuration options](configs/README.md) for a description of configuration options


## Models
<Will be added soon>

## Reference
*Self-Supervised Nuclei Segmentation in Histopathological Images Using Attention*, Mihir Sahasrabudhe, Stergios Christodoulidis, Roberto Salgado, Stefan Michiels, Sherene Loi, Fabrice Andre, Nikos Paragios, Maria Vakalopoulou, MICCAI 2020

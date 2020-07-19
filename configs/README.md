## Configuration options


### Model options

```model_arch``` : ```nn.Module``` to be used for the attention net ```F``` and the feature extraction of the scale net ```G```. Must return a dictionary with ```'attention'``` and ```'fg_feats'``` as keys, which represent, respectively, the attention output of ```F```, and the feature vector from ```G```
```scale_arch``` : ```nn.Module``` to be used to classify ```fg_feats``` obtained from ```G```
```freeze_model``` : Whether to freeze the attention and scale networks or not
```ndf``` : Width parameter for ```model_arch```
```nc``` : Number of channels in input images
```attention_sparsity``` : The desired sparsity. Specified as a real number in ```[0, 1]```
```attention_sparsity_r``` : Parameter determining the compression in the sigmoid

### Training options

```batch_size``` : Training batch size to use
```lr``` : Desired learning rate
```momentum``` : Desired momentum for optimiser
```weight_decay``` : L2 regularisation weight hyperparameter for optimiser
```n_epochs``` : Number of training epochs
```losses``` : A list specifying which losses to use for training. See variable ```loss_definitions``` in ```train.py``` for possible losses
```predictions``` : A list specifying what predictions to make
```checkpoint_special``` : A list or an int. A list specifies which epochs to keep specific checkpoints. An int specifies a checkpoint every so many epochs
```lr_decay``` : Decay factor for learning rate
```lr_decay_every``` : LR is decayed at ...
```equivariance_scale``` : Whether to enforce equivariance to scale for attention network ```F```. Preferably ```False```
```equivariance_aug``` : Whether to enforce equivariance to rigid transforms for attention network ```F```. See ```utils.py``` for possible values for this option
```optimiser``` : Which optimiser to use. Allowed values are ```'adam'``` and ```'sgd'```. 
```pixel_means``` : Pixel means for input images
```pixel_stds``` : Pixel stds for input images
```max_train_batches_per_epoch``` : Number of maximum training batches per epoch
```max_val_batches_per_epoch``` : Number of maximum validation batches per epoch
```use_colour_transform``` : Whether to use colour augmentation 
```use_image_transforms``` : Whether to use geometric augmentation 

### Dataset options

```dset_name``` : Name of dataset
```image_size``` : Sizes of input images. If an image is not this size, it is padded by placing the original in the top-left corner
```patch_size``` : Sizes of patches to pick out from input images
```workers``` : Number of dataset workers
```dataroot``` : Path to dataset files
```stain_normaliser_file``` : Relative path to stain normalisation target (relative to ```dataroot```)
```hed_decomp``` : Whether to use HED decomposition instead of RGB images
```hed_channels``` : Which HED channels to use
```seg_threshold``` : Threshold to reject input tiles. This value in ```[0, 1]``` specifies at least what fraction of an input image must be tissue in order for it to be used
```levels``` : Magnification levels to use for training. ```OpenSlide```'s convention is followed, so ```'max'``` denotes the maximum magnification, while ```'-1'``` and ```'-2'``` denote one and two lower levels of magnification, respectively. 
```splits_file``` : Splits file specifying train/val/test split in ```dataroot```.

### Initialisation options

```load``` : Initialise entire expriment from this path
```init_model``` : Initialise only the model from this path

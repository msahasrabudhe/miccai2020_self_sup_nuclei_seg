import  torch
import  torch.nn.functional as      F
import  sys
import  numpy               as      np
from    attr_dict           import  *

# ======================================================================
#   Some useful global variables
PROOT                               = 'PROOT'
PT                                  = 'PT'
SEG_COVER                           = 'SEG_COVER'

# Patient attributes. 
MID                                 = 'MID'

# Macros for dataset names. 
MONUSEG                             = 'monuseg'
MONUSEGWSI                          = 'monuseg_wsi'
PATCHCAMELYON                       = 'patchcamelyon'
ACCEPTED_DATASETS                   = [MONUSEG, MONUSEGWSI]

LEVEL                               = 'level'
BCE_LOSS                            = 'bce'
CE_LOSS                             = 'ce'
BCE_NOLOGITS_LOSS                   = 'bce_nl'
POS_TARGET                          = np.log(2 + np.sqrt(3)).astype(np.float32)
NEG_TARGET                          = np.log(2 - np.sqrt(3)).astype(np.float32)
NORM_FACTOR                         = 1./np.sqrt(2 * np.pi).astype(np.float32)

L_SMOOTHNESS                        = 'smooth'
L_SPARSITY                          = 'sparsity'
L_SPARSITY_L1                       = 'sparsity_l1'
L_SCALE                             = 'scale'

SKIP_ACCURACY_LOSSES                = [L_SMOOTHNESS, L_SPARSITY, L_SPARSITY_L1]
EPS                                 = 0.00000001


BKWD_CMPTBL_DICT = {
    'model_arch'                    :   'Dilated10ConvAttentionMap1x1AvgTauResNet34WithSparsity2',
    'scale_arch'                    :   'LinearScaleClassifier',
    'projection_multiplier'         :   -1,
    'attention_feats_ex_ndf'        :   64,
    'attention_sparsity'            :   0.1,
    'attention_sparsity_r'          :   1.0,
    'freeze_model'                  :   False,

    'equivariance_scale'            :   False,
    'equivariance_aug'              :   [],

    'init_model'                    :   '',
    'use_colour_transform'          :   False,
    'use_image_transforms'          :   True,
    'use_global_transform'          :   False,

    'optimiser'                     :   'sgd',
    'predictions'                   :   [L_SCALE],

    'n_imgs_per_class'              :   [1.0, 1.0, 1.0],
    'max_train_batches_per_epoch'   :   -1,
    'max_val_batches_per_epoch'     :   -1,

    'adj_overlap'                   :   16,
    'pixel_means'                   :   [0.5, 0.5, 0.5],
    'pixel_stds'                    :   [0.5, 0.5, 0.5],

    'dset_name'                     :   'monuseg_wsi',
    'maskroot'                      :   False,
    # Extra dataset options. 
    'stain_normaliser_file'         :   False,
}

"""
Custom function to print options in a formatted manner. 
"""
def print_config(options, prefix=''):
    for key in options:
        if isinstance(options[key], AttrDict):
            write_flush(prefix+'{}:\n' %(key))
            print_config(options[key], prefix+'\t')
        else:
            write_flush(prefix+'{:30s}: {}\n'.format(key, options[key]))
    return 



def fix_backward_compatibility(options, cmptbl_dict=BKWD_CMPTBL_DICT):
    for key in cmptbl_dict:
        val         = cmptbl_dict[key]
        if key not in options:
            if isinstance(val, dict):
                fix_backward_compatibility(options[key], cmptbl_dict=cmptbl_dict[key])
            else:
                options[key] = val
#    return options

def group_params(models_list):
    for m in models_list:
        for p in m.parameters():
            yield p


def load_state(path):
    return torch.load(path, map_location=lambda s,l:s)

def write_flush(text, stream=sys.stdout):
    stream.write(text)
    stream.flush()
    return


def align_left(text):
    write_flush('%-70s' %(text))
    return

def write_okay():
    write_flush('[  OK  ]\n')
    return


def trim_state_dict(complete_dict, trim_key):
    """
    Trim state dict so that only those keys starting with trim_key
    remain, and the prefixed module name is removed from the key name.
    """
    trimmed     = {
                    k.replace(trim_key+'.', ''):complete_dict[k] 
                    for k in complete_dict if k.startswith(trim_key)
                  } 
    
    # The resulting dictionary can be used to load a part of a model. 
    return trimmed

# ===============================================================================================================================

def l1_smoothness_loss(img):
    # L1 smoothing on spatial gradients. 
    row_smooth          = torch.mean(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    col_smooth          = torch.mean(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    return row_smooth + col_smooth

def l2_smoothness_loss(img):
    # L2 smoothing on spatial gradients. 
    row_smooth          = torch.norm(img[:, :, :-1, :] - img[:, :, 1:, :], p=2)
    col_smooth          = torch.norm(img[:, :, :, :-1] - img[:, :, :, 1:], p=2)
    return row_smooth + col_smooth

def masked_l2_smoothness_loss(img, mask):
    # L2 smoothing on spatial gradients. 
    row_smooth          = torch.norm((img[:, :, :-1, :] - img[:, :, 1:, :]) * mask[:, :, 1:, :], p=2)
    col_smooth          = torch.norm((img[:, :, :, :-1] - img[:, :, :, 1:]) * mask[:, :, :, 1:], p=2)
    return row_smooth + col_smooth

# ===============================================================================================================================
def positive_saliency(grad):
    return F.relu(grad) / grad.max()

def negative_saliency(grad):
    return F.relu(-1 * grad) / (-1 * grad).max()

# ===============================================================================================================================

def entropy(logits):
    """
    Compute entropy of the probability distribution given by the logits. 
    """
    probs               = F.softmax(logits, dim=1)
    log_probs           = F.log_softmax(logits, dim=1)
    
    return torch.sum(probs * log_probs, dim=1)
           


# ===============================================================================
def soft_dice_loss(y_pred, y_true):
    smooth                  = 1.

    iflat                   = y_pred.contiguous().view(-1)
    tflat                   = y_true.contiguous().view(-1)
    intersection            = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
                          (iflat.sum() + tflat.sum() + smooth))

# ===============================================================================================================================
def stitch_images(img_dict, lib='th', ch=None, overlap=0):
    """
    img_dict contains images, their locations specified as col_row
    """

    locations           = list(img_dict.keys())
   

    min_rows            = min([int(l.split('_')[1]) for l in locations])
    min_cols            = min([int(l.split('_')[0]) for l in locations])
    max_rows            = max([int(l.split('_')[1]) for l in locations])
    max_cols            = max([int(l.split('_')[0]) for l in locations])

    ov                  = overlap // 2

    if lib == 'th':
        img_size        = img_dict[locations[0]].shape[-1]
        n_ch            = img_dict[locations[0]].size(-3) if ch is None else 1
        zero_img        = torch.zeros(n_ch, img_size - 2*ov, img_size - 2*ov)
        cat_func        = lambda L, d: torch.cat(L, dim=d)
        if ov > 0:
            patch_func  = lambda img: img[:, ov:-ov, ov:-ov]
        else:
            patch_func  = lambda img: img
    elif lib == 'np':
        img_size        = img_dict[locations[0]].shape[0]
        n_ch            = img_dict[locations[0]].shape[-1] if ch is None else 1
        zero_img        = np.zeros((img_size - 2*ov, img_size - 2*ov, n_ch), dtype=img_dict[locations[0]].dtype)
        cat_func        = lambda L, d: np.concatenate(L, axis=d-1)
        if ov > 0:
            patch_func  = lambda img: img[ov:-ov, ov:-ov, :]
        else:
            patch_func  = lambda img: img
    
    for r in range(min_rows, max_rows+1):
        for c in range(min_cols, max_cols+1):
            loc         = '%d_%d' %(c, r)
            if loc in img_dict:
                this_   = patch_func(img_dict[loc])
            else:
                this_   = zero_img

            row_        = this_ if c == min_cols else cat_func([row_, this_], -1)

        stitched_       = row_ if r == min_rows else cat_func([stitched_, row_], -2)

    return stitched_


# ===============================================================================
#   Define geometric transformation losses. 
#   Tensors here are defined as B x C x H x W
#

def AugTransform_flipX(T):
    # Flip the images in tensor T along X.
    return torch.flip(T, (3,))

def AugTransform_flipY(T):
    # Flip the images in tensor T along Y.
    return torch.flip(T, (2,))

def AugTransform_transpose(T):
    # Transpose the image. 
    return T.permute(0,1,3,2)

def AugTransform_rot90(T):
    # Rotate the image 90 degrees clockwise. 
    return AugTransform_transpose(AugTransform_flipX(T))

def AugTransform_rot180(T):
    # Rotate the image 180 degrees. 
    return AugTransform_flipX(AugTransform_flipY(T))

def AugTransform_rot270(T):
    # Rotate the image 270 degrees. 
    return AugTransform_transpose(AugTransform_flipY(T))


def AugTransform_D2(T):
    # Downsample the image by a factor of 2. 
    return F.interpolate(T, scale_factor=0.5, mode='bilinear', align_corners=False)

def AugTransform_D4(T):
    # Downsample the image by a factor of 4. 
    return F.interpolate(T, scale_factor=0.25, mode='bilinear', align_corners=False)

def AugTransform_U2(T):
    # Upsample the image by a factor of 2. 
    return F.interpolate(T, scale_factor=2, mode='bilinear', align_corners=False)

def AugTransform_U4(T):
    # Upsample the image by a factor of 4. 
    return F.interpolate(T, scale_factor=4, mode='bilinear', align_corners=False)


AUG_TRANSFORMS_DICT     = {
            'ID'            : {
                                'forward'   :   lambda x: x,
                                'backward'  :   lambda x: x,
                              },
            'FX'            : {
                                'forward'   :   AugTransform_flipX,
                                'backward'  :   AugTransform_flipX,
                              },
            
            'FY'            : {
                                'forward'   :   AugTransform_flipY,
                                'backward'  :   AugTransform_flipY,
                              },

            'TR'            : {
                                'forward'   :   AugTransform_transpose,
                                'backward'  :   AugTransform_transpose,
                              },
            
            'R90'           : {
                                'forward'   :   AugTransform_rot90,
                                'backward'  :   AugTransform_rot270,
                              },

            'R180'          : {
                                'forward'   :   AugTransform_rot180,
                                'backward'  :   AugTransform_rot180,
                              },

            'R270'          : {
                                'forward'   :   AugTransform_rot270,
                                'backward'  :   AugTransform_rot90,
                              },

            'D2'            : {
                                'forward'   :   AugTransform_D2,
                                'backward'  :   AugTransform_U2,
                              },

            'D4'            : {
                                'forward'   :   AugTransform_D4,
                                'backward'  :   AugTransform_U4,
                              },

            'U2'            : {
                                'forward'   :   AugTransform_U2,
                                'backward'  :   AugTransform_D2,
                              },

            'U4'            : {
                                'forward'   :   AugTransform_U4,
                                'backward'  :   AugTransform_D4,
                              },
}


    

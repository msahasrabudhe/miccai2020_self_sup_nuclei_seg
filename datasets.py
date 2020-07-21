import  torch
from    torch.nn                import  functional  as F
import  torch.utils.data
import  torchvision.utils       as      vutils
import  torchvision.transforms  as      vtransforms

# For stain normalisation. 
from    colorTransferCV2        import  StainNormalizerLAB, StainNormalizerL
#import staintools

from    attr_dict               import  *
from    utils                   import  *
from    IPython                 import  embed

import  pickle
import  matplotlib.pyplot       as      plt
import  skimage.color           as      skcolor
from    PIL                     import  Image
import  random
import  numpy                   as      np
import  os
import  imageio
import  sys
import  scipy.io
import  skimage.color           as      skcolour

# Workaround to allow non-zero dataloader workers. 
import h5py

# To handle annotations CSV file. 
import pandas


#   ====================================================================================================
def pil_loader(path):
    img = Image.open(path)
    return img
#   ====================================================================================================

# ======================================================================================================
def NormaliseAndFixImageSize(patch_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    """
    NormaliseAndFixImageSize ::: Normalises a PIL image according to given mean and std. 
    Also fixes image size so that it measures patch_size x patch_size. 
    """
    return vtransforms.Compose([
             vtransforms.ToTensor(),
             vtransforms.Normalize(mean, std),
             FixImageSize(patch_size, pad_value=0),
           ])

def ColourTransform(mean=1.0, std=0.03):
    """
    ColourTransform ::: Adds data augmentation by breaking the image apart 
    into H & E stains, and randomly modifying their concentration. 
    """
    def transform(img):
        hed         = skcolour.rgb2hed(img / 255.0)
        alphas      = np.random.normal(size=(1,1,3), loc=mean, scale=std)
        hed         = hed * alphas
        img         = skcolour.hed2rgb(hed).astype(np.float32)
        return img

    return transform
# ======================================================================================================


#   ====================================================================================================
def NormaliseTransform(image_size, patch_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return vtransforms.Compose([
             vtransforms.ToTensor(),
             vtransforms.Normalize(mean, std),
             FixImageSize(image_size, pad_value=0),
             RandomCropTensor(patch_size),
           ])
#   ====================================================================================================

def ToTorchTransform(image_size, patch_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return vtransforms.Compose([
             vtransforms.ToTensor(),
             vtransforms.Normalize(mean, std),
             FixImageSize(image_size, pad_value=0),
             RandomCropTensor(patch_size),
           ])
#   ====================================================================================================



def NormaliseTransformNoFixImageSize(image_size, patch_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return vtransforms.Compose([
             vtransforms.ToTensor(),
             vtransforms.Normalize(mean, std),
             RandomCropTensor(patch_size),
           ])
#   ====================================================================================================



def NormaliseTransformWithoutPatchSize(image_size, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return vtransforms.Compose([
             vtransforms.ToTensor(),
             vtransforms.Normalize(mean, std),
             FixImageSize(image_size, pad_value=0),
           ])
#   ====================================================================================================


#   ====================================================================================================
def NormaliseTransformNoPad(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    return vtransforms.Compose([
             vtransforms.ToTensor(),
             vtransforms.Normalize(mean, std),
           ])
#   ====================================================================================================

class RandomNoiseTransform(object):
    def __init__(self, options, mean=0.0, std=1.0, fixed=True):
        self.mean           = mean
        self.std            = std
        self.fixed          = fixed
        self.nc             = options.nc
        if fixed:
            self.noise      = np.random.normal(size=(1,1,self.nc), loc=mean, scale=std).astype(np.float32)
        else:
            self.noise      = None
    def __call__(self, img):
        if self.fixed:
            return img + self.noise
        return img + np.random.normal(size=(1,1,self.nc), loc=self.mean, scale=self.std).astype(np.float32)

# ======================================================================================================
class RandomRotation(object):
    """
    A randomly chosen rotation is applied to a PIL image.
    """
    def __init__(self, angles_list):
        self.angles_list    = angles_list
    def __call__(self, img):
        A                   = np.random.choice(self.angles_list)
        return img.rotate(A)
# ======================================================================================================

class StackTransforms(object):
    """
    A transformation class which applies a list of transforms
    to an image, and returns a stack with each of 
    the transforms applied to the image.
    """
    
    def __init__(self, transforms):
        self.transforms = transforms
        
    def __call__(self, img):
        results = []
        for t_ in self.transforms:
            if t_ is None:
                results.append(img)
            else:
                results.append(t_(img))
        return results

#   ====================================================================================================
def ColourTransformSampleEvery(mean=1.0, std=0.03):
    def transform(img):
        # Colour augmentation by breaking the image apart into H & E stains, and modifying their concentration. 
        hed         = skcolor.rgb2hed(img / 255.0)
        alphas      = np.random.normal(size=(1,1,3), loc=mean, scale=std)
        hed         = hed * alphas
        img         = skcolor.hed2rgb(hed).astype(np.float32)
        return img

    return transform
#   ====================================================================================================


#   ====================================================================================================
def ColourTransformFix(mean=1.0, std=0.03):
    alphas          = np.random.normal(size=(1,1,3), loc=mean, scale=std)
    def transform(img):
        # Colour augmentation by breaking the image apart into H & E stains, and modifying their concentration. 
        hed         = skcolor.rgb2hed(img / 255.0)
        hed         = hed * alphas
        img         = skcolor.hed2rgb(hed).astype(np.float32)
        return img

    return transform
#   ====================================================================================================

    

#   ====================================================================================================
# A transformation to pad an image so that every image can be ensured to be of the same size. 
# The padding is done so that the patch rests in the top-left corner. 
class FixImageSize(object):
    def __init__(self, size, pad_value=0):
        if isinstance(size, int):
            self.size       = (size, size)
        elif isinstance(size, list) and len(size) == 2 and all([isinstance(s, int) for s in size]):
            self.size       = size
        else:
            raise ValueError('Expected either an integer or a list of two integers as the first argument to FixImageSize. Got {}'.format(size))

        self.pad_value      = pad_value

    def __call__(self, image):
        size_x              = image.size(2)
        size_y              = image.size(1)

        pad_list            = [0, 0, 0, 0]
        if size_x < self.size[1]:
            pad_list[1]     = self.size[1] - size_x
        if size_y < self.size[0]:
            pad_list[3]     = self.size[0] - size_y
        
        image               = F.pad(image, pad_list, 'constant', self.pad_value)
        return image
#   ====================================================================================================


#   ====================================================================================================
# A transformation that randomly crops a tensor, but fixed parameters once, so that it be used on 
#   multiple images. 
class RandomCropTensorFixed(object):
    def __init__(self, image_size=None, patch_size=None):
        if isinstance(image_size, int):
            self.i_size     = (image_size, image_size)
        elif isinstance(image_size, list) and len(image_size) == 2 and all([isinstance(s, int) for s in image_size]):
            self.i_size     = image_size
        else:
            raise ValueError('Expected either an integer or a list of two integers as the first argument to RandomCropTensor. Got {}'.format(image_size))

        if isinstance(patch_size, int):
            self.p_size     = (patch_size, patch_size)
        elif isinstance(patch_size, list) and len(patch_size) == 2 and all([isinstance(s, int) for s in patch_size]):
            self.p_size     = patch_size
        else:
            raise ValueError('Expected either an integer or a list of two integers as the first argument to RandomCropTensor. Got {}'.format(patch_size))

        self.start_row      = int(np.random.randint(self.i_size[0] - self.p_size[0]))
        self.start_col      = int(np.random.randint(self.i_size[1] - self.p_size[1]))
        self.end_row        = self.start_row + self.p_size[0]
        self.end_col        = self.start_col + self.p_size[1]

    def __call__(self, image):
        return image[:, self.start_row:self.end_row, self.start_col:self.end_col]

#   ====================================================================================================



#   ====================================================================================================
# A transformation that randomly crops a tensor
class RandomCropTensor(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size       = (size, size)
        elif isinstance(size, list) and len(size) == 2 and all([isinstance(s, int) for s in size]):
            self.size       = size
        else:
            raise ValueError('Expected either an integer or a list of two integers as the first argument to RandomCropTensor. Got {}'.format(size))

    def __call__(self, image):
        nc, rows, cols      = image.size()
        if rows == self.size[0] and cols == self.size[1]:
            return image

        start_row           = int(np.random.randint(rows - self.size[0]))
        start_col           = int(np.random.randint(cols - self.size[1]))
        end_row             = start_row + self.size[0]
        end_col             = start_col + self.size[1]

        return image[:, start_row:end_row, start_col:end_col]

#   ====================================================================================================


#   ====================================================================================================
class MoNuSegWSIImageset(torch.utils.data.Dataset):
    def __init__(self, 
                 options, 
                 splits=['train'], 
                 img_transforms=None, 
                 force_mid=None, 
                 force_levels=None, 
                 force_limits=None):
        """
        options         AttrDict        Specifies experiment options
        splits          list            Which splits to include. Each value in 
                                        this list must have a corresponding
                                        entry in the splits' YAML file. 
        img_transforms  object or None  A set of transforms to apply on 
                                        images
        force_mid       str             Only return images belonging to this
                                        particular ID. 
        force_levels    list            A list of levels to override the one in options. 
        force_limits    list            A list of four numbers specifying the 
                                        extremeties of a subsection to be extracted. 
                                        The entries are [left, top, width, height]
        """

        super(MoNuSegWSIImageset, self).__init__()

        self.dataroot               = options.dataroot

        # Do not add colour transform if not training. 
        self.if_train               = 'train' in splits

        self.threshold              = options.seg_threshold

        self.use_global_transform   = options.use_global_transform
        self.use_colour_transform   = options.use_colour_transform
        self.hed_decomp             = options.hed_decomp
        self.hed_channels           = options.hed_channels


        self.force_limits           = force_limits
        if self.force_limits is not None:
            l_left, l_top, l_w, l_h = self.force_limits

        if options.stain_normaliser_file:
#            with open(os.path.join(options.dataroot, options.stain_normaliser_file), 'rb') as fp:
#                self.stain_normaliser   = pickle.load(fp)
            stain_norm_target       = imageio.imread(os.path.join(options.dataroot, options.stain_normaliser_file))
            self.stain_normaliser   = StainNormalizerLAB()
            self.stain_normaliser.fit(stain_norm_target)
        else:
            self.stain_normaliser   = False


        assert len(self.hed_channels) == options.nc, 'HED decomposition specified with channels\
'+str(self.hed_channels)+' but number of input channels for networks is %d.'%(options.nc)

        self.levels                 = force_levels if force_levels is not None else options.levels
        self.f_levels               = []
        for l in self.levels:
            if l == 'max':
                self.f_levels.append(l)
            else:
                self.f_levels.append(int(l))

        self.image_size             = options.image_size
        if options.patch_size == -1: # or not self.if_train:
            # Use entire image if validation phase. 
            self.patch_size         = options.image_size
        else:
            self.patch_size         = options.patch_size

        self.splits_file            = os.path.join(self.dataroot, options.splits_file)
        self.seg_cover_file         = os.path.join(self.dataroot, options.seg_cover_file)
        assert os.path.exists(self.splits_file),    'Specified splits_file {} does not exist!'.format(self.splits_file)
        assert os.path.exists(self.seg_cover_file), 'Specified seg_cover_file {} does not exist!'.format(self.seg_cover_file)

        if force_mid is None:
            self.splits             = load_yaml(self.splits_file)
            self.mids               = []
            for ss in splits:
                self.mids          += self.splits[ss]
            # Convert all mids to string. 
            self.mids               = [str(m) for m in self.mids]
        else:
            self.mids               = [str(force_mid)]

        with open(self.seg_cover_file, 'rb') as fp:
            self.seg_cover          = pickle.load(fp)

        self.img_transforms         = img_transforms
        
        # Fix random perturbation in colour transform if force_mid is not None. We want to use the same perturbation across 
        #   all tiles in an image if training with whole patch.
        if self.hed_decomp:
            if force_mid is not None:
                self.colour_transform = RandomNoiseTransform(options, mean=0, std=0.03, fixed=True) 
            else:
                self.colour_transform = RandomNoiseTransform(options, mean=0, std=0.03, fixed=False)
            self.normalise          = NormaliseTransform(self.image_size, self.patch_size, mean=options.pixel_means, std=options.pixel_stds)
        else:
            if force_mid is not None:
                self.colour_transform = ColourTransformFix(mean=1.0, std=0.03)
            else:
                self.colour_transform = ColourTransformSampleEvery(mean=1.0, std=0.03)
            self.normalise          = NormaliseTransform(self.image_size, self.patch_size, mean=options.pixel_means, std=options.pixel_stds)
#            self.normalise      = NormaliseTransformNoPad(mean=options.pixel_means, std=options.pixel_stds)
   
        n_l_files                   = [0 for f in range(len(self.f_levels))]


        
        dataset_info_pickle_file    = os.path.join(options.output_dir, 'dataset_' + '_'.join(splits) + '.pkl')

        if all([x is None for x in [force_mid, force_levels, force_limits]]) \
            and os.path.exists(dataset_info_pickle_file):
            print('Found dataset info at %s. Loading saved info ...' %(dataset_info_pickle_file))
            with open(dataset_info_pickle_file, 'rb') as fp:
                dataset_info        = pickle.load(fp)
                self.file_list      = dataset_info['file_list']
                self.mid_list       = dataset_info['mid_list']
                self.level_list     = dataset_info['level_list']
                self.olevel_list    = dataset_info['olevel_list']
                self.class_indices  = dataset_info['class_indices']
        else:
            self.file_list          = []
            self.mid_list           = []
            self.level_list         = []
            self.olevel_list        = []
    
            # Since the classes are stored sequentially (according to level), we store
            # the indices where all images of a class end. 
            self.class_indices      = []
               
            for f_, fl in enumerate(self.f_levels, 0):
                for mid in self.mids:
                    mid_root        = os.path.join(self.dataroot, mid, 'slide_files/')
                    levels_list     = [int(x) for x in os.listdir(mid_root)]
        
                    max_level       = max(levels_list)
    
                    if fl == 'max':
                        nl          = str(max_level)
                    elif fl < 0:
                        nl          = str(max_level + fl)
                    else:
                        nl          = str(fl)
    
               
                    slide_root      = os.path.join(self.dataroot, mid, 'slide_files/', nl)
                    files_          = []
                    for f in os.listdir(slide_root):
                        if f not in self.seg_cover[mid][nl]:
                            print('mid: {}, nl: {}, f: {}'.format(mid, nl, f))
                        elif self.seg_cover[mid][nl][f] > self.threshold:
                            if self.force_limits is None:
                                files_.append(f)
                            else:
                                f_col   = int(f.split('.')[0].split('_')[0])
                                f_row   = int(f.split('.')[0].split('_')[1])
                                if f_col >= l_left and f_col <= l_left + l_w and\
                                   f_row >= l_top  and f_row <= l_top  + l_h:
                                    files_.append(f)
#                                    print('Including file %s because %d \\in [%d, %d] and %d \\in [%d, %d]' 
#                                           %(f, f_col, l_left, l_left + l_w, f_row, l_top, l_top + l_h))
                                else:
#                                    print('Excluding file %s because %d \\notin [%d, %d] and %d \\notin [%d, %d]' 
#                                           %(f, f_col, l_left, l_left + l_w, f_row, l_top, l_top + l_h))
                                    pass
                                
                    mids_           = [mid for f in files_]
                    levels_         = [nl for f in files_]
                    olevels_        = [f_ for x in files_]
    
                    self.file_list  += files_
                    self.mid_list   += mids_
                    self.level_list += levels_
                    self.olevel_list += olevels_
    
                    n_l_files[f_] += len(files_)
    
                self.class_indices.append(len(self.file_list))
            
            dataset_info        = {}
            dataset_info['file_list']       = self.file_list
            dataset_info['mid_list']        = self.mid_list
            dataset_info['level_list']      = self.level_list
            dataset_info['olevel_list']     = self.olevel_list
            dataset_info['class_indices']   = self.class_indices

            if all([x is None for x in [force_mid, force_levels, force_limits]]):
                with open(dataset_info_pickle_file, 'wb') as fp:
                    pickle.dump(dataset_info, fp)
                print('Wrote dataset info to %s.' %(dataset_info_pickle_file))

    def __getitem__(self, index):
        mid         = self.mid_list[index]
        img_name    = self.file_list[index]
        level       = self.level_list[index]
        olevel      = self.olevel_list[index]

        file_path   = os.path.join(self.dataroot, mid, 'slide_files/', level, img_name)
        img         = pil_loader(file_path)

        if self.img_transforms is not None:
            img     = self.img_transforms(img)

        if isinstance(img, list):   
            # In case we used a transforms that gives several images from one. 
            stack       = [np.array(img_) for img_ in img]
            # Stain normalisation. 
            if self.stain_normaliser:
                stack = [self.stain_normaliser(img_) for img_ in stack]

            # H&E decomposition. 
            if self.hed_decomp:
                stack   = [skcolor.rgb2hed(img_ / 255.0).astype(np.float32) for img_ in stack]
                stack   = [img_[:,:,self.hed_channels] for img_ in stack]

            if self.use_colour_transform and self.if_train:
                stack   = [self.colour_transform(img_) for img_ in stack]
            stack       = [self.normalise(img_) for img_ in stack]
            img         = torch.stack(stack)
        else:
            # Convert to Numpy array from PIL Image. 
            img         = np.array(img)
            # Stain normalisation. 
            if self.stain_normaliser:
                img     = self.stain_normaliser(img)

            # H&E decomposition. 
            if self.hed_decomp:
                img     = skcolor.rgb2hed(img / 255.0).astype(np.float32) 
                img     = img[:,:,self.hed_channels]
    
            if self.use_colour_transform and self.if_train:
                img     = self.colour_transform(img)
    
            img         = self.normalise(img)

        # Added img_name[:-4] on 2019-11-30 to accomodate for 
        # adjacency graph. 
        return (img, olevel, img_name[:-4], img_name[:-4]) 
    

    def __len__(self):
        return len(self.file_list)
#   ====================================================================================================


class MoNuSegImageset(torch.utils.data.Dataset):
    def __init__(self, options, splits=['train'], img_transforms=None, force_id=None, add_transforms=True, n_folds=70):
        super(MoNuSegImageset, self).__init__()
        
        self.options                = options
        self.image_size             = options.image_size
        self.patch_size             = options.patch_size
        self.pixel_means            = options.pixel_means
        self.pixel_stds             = options.pixel_stds
        self.use_colour_transform   = options.use_colour_transform
        self.if_train               = 'train' in splits
       
        if force_id:
            self.ids                = [force_id]
        else:
            self.ids                = []
            dataset_split           = load_yaml(os.path.join(options.dataroot, options.splits_file))
            for s in splits:
                self.ids           += dataset_split[s]

        self.n_ids                  = len(self.ids)

        self.levels                 = options.levels
        self.n_levels               = len(self.levels)

        self.images                 = []
        self.masks                  = []

        for level_ in self.levels:
            for id_ in self.ids:
                img_path            = os.path.join(options.dataroot, level_, id_+'.png')
                mask_path           = os.path.join(options.maskroot, level_, id_+'.mat')
#                img                 = pil_loader(img_path)
                self.images.append(img_path)
                self.masks.append(mask_path)

        self.img_transforms         = img_transforms        
        if force_id is not None:
            self.colour_transform   = ColourTransformFix(mean=1.0, std=0.03)
        else:
            self.colour_transform   = ColourTransformSampleEvery(mean=1.0, std=0.03)
        self.normalise              = NormaliseTransformNoPad(mean=options.pixel_means, std=options.pixel_stds)
        self.crop                   = RandomCropTensorFixed(image_size=self.image_size, patch_size=self.patch_size)

        # This option adds transforms to the image as well as the mask.
        self.add_transforms         = add_transforms
#
        if options.stain_normaliser_file:
#            with open(os.path.join(options.dataroot, options.stain_normaliser_file), 'rb') as fp:
#                self.stain_normaliser   = pickle.load(fp)
            stain_norm_target       = imageio.imread(os.path.join(options.dataroot, options.stain_normaliser_file))
            self.stain_normaliser   = StainNormalizerLAB()
            self.stain_normaliser.fit(stain_norm_target)
        else:
            self.stain_normaliser   = False

        # Artificially augment length of the dataset. 
        self.n_folds                = n_folds
        self.true_length            = len(self.images)          # Also equal to self.n_ids * self.n_levels
        self.length                 = self.true_length * self.n_folds


    def image_id_and_level_from_index(self, index):
        level_                      = index // self.n_ids
        id_                         = index % self.n_ids

        return id_, level_

    def index_from_image_id_and_level(self, id_, level_):
        return self.n_ids * level_ + self.id_
                

    def __getitem__(self, index):
        # True index is index mod self.true_length
        index                       = index % self.true_length

        id_, level_                 = self.image_id_and_level_from_index(index)

        img                         = pil_loader(self.images[index])
        mask                        = scipy.io.loadmat(self.masks[index])
        mask                        = Image.fromarray(np.uint8(mask['indiv_mask'] > 0))

        # Choose custom image transforms.
        if self.add_transforms:
            chosen_rotation         = np.random.choice([0, 90, 180, 270])
            additional_transforms   = []
            # Choose random flip
            additional_transforms.append(RandomRotation([chosen_rotation]))
            if np.random.rand() > 0.5:
                additional_transforms.append(vtransforms.RandomHorizontalFlip(p=1))
            if np.random.rand() > 0.5:
                additional_transforms.append(vtransforms.RandomVerticalFlip(p=1))

        additional_transforms       = vtransforms.Compose(additional_transforms)
        if self.add_transforms:
            img                     = additional_transforms(img)
            mask                    = additional_transforms(mask)

        if self.img_transforms:
            img                     = self.img_transforms(img)

        if isinstance(img, list):   
            # In case we used a transforms that gives several images from one. 
            stack                   = [np.array(img_) for img_ in img]
            if self.use_colour_transform and self.if_train:
                stack               = [self.colour_transform(img_) for img_ in stack]
            stack                   = [self.normalise(img_) for img_ in stack]
            stack                   = [self.crop(img_) for img_ in stack]
            img                     = torch.stack(stack)
        else:
            # Convert to Numpy array from PIL Image. 
            img                     = np.array(img)
   
            # Stain normalisation. 
            if self.stain_normaliser:
                img     = self.stain_normaliser(img)

            if self.use_colour_transform and self.if_train:
                img                 = self.colour_transform(img)
    
            img                     = self.normalise(img)
            img                     = self.crop(img)

            mask                    = torch.from_numpy(np.array(mask)).float().unsqueeze(0)
            mask                    = self.crop(mask)

        # Added img_name[:-4] on 2019-11-30 to accomodate for 
        # adjacency graph. 
        return (img, level_, mask, self.ids[id_])
 

    def __len__(self):
        return self.length

#   ====================================================================================================

class MoNuSegWSIScaleEqualSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset    = dataset
        self.n_classes  = len(dataset.class_indices)

        self.class_ids  = [0] + dataset.class_indices
        self.n_imgs_per_class   = [self.class_ids[i] - self.class_ids[i-1] for i in range(1, self.n_classes+1)]

        self.n_imgs_to_sample   = min(self.n_imgs_per_class)

        self.shuffle()

    def shuffle(self):
        self.shuffled_img_ids   = []
        for c in range(self.n_classes):
            start_id            = self.class_ids[c]
            end_id              = self.class_ids[c+1]
            perm                = start_id + np.random.permutation(end_id - start_id)
            # Choose only the first n_imgs_to_sample
            perm                = perm[:self.n_imgs_to_sample]

            self.shuffled_img_ids += perm.tolist()

        self.shuffled_img_ids   = np.random.permutation(self.shuffled_img_ids).tolist()

    def __iter__(self):
        return iter(self.shuffled_img_ids)

    def __len__(self):
        return len(self.shuffled_img_ids)
#   ====================================================================================================


class MoNuSegScaleEqualSampler(torch.utils.data.Sampler):
    def __init__(self, dataset):
        self.dataset            = dataset
        self.n_classes          = dataset.n_levels
        self.n_ids              = dataset.n_ids

        self.class_ids          = []
        for c_ in range(self.n_classes):
            c_ids_              = range(self.n_ids * c_, self.n_ids * (c_ + 1))
            self.class_ids.append(c_ids_)

        # HACK: Generate as many indices as the number of batches
        #   multiplied by the batch size. 
        if dataset.options.max_train_batches_per_epoch == -1:
            max_n_batches       = len(dataset) // dataset.options.batch_size + 1
        else:
            max_n_batches       = dataset.options.max_train_batches_per_epoch
        
        self.n_indices          = max_n_batches * dataset.options.batch_size

        self.shuffle()
        
    def shuffle(self):
        self.indices            = [ random.choice(self.class_ids[random.choice(range(self.n_classes))]) \
                                   for i in range(self.n_indices) ]
        return

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return self.n_indices

#   ====================================================================================================

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, options, ext='.tif', dataroot=None, filenames=None):
        super(TestDataset, self).__init__()
        
        self.options            = options
        self.dataroot           = dataroot
        self.im_root            = self.dataroot

        if not ext.startswith('.'):
            ext                 = '.' + ext

        self.ext                = ext

        if not filenames:
            self.files          = [p.replace(self.ext,'') for p in os.listdir(self.im_root)]
        else:
            self.files          = filenames

        self.stds               = options.pixel_stds
        self.means              = options.pixel_means
        
        if options.stain_normaliser_file:
            target              = imageio.imread(os.path.join(options.dataroot, options.stain_normaliser_file))
            self.stain_normaliser = StainNormalizerLAB()
            self.stain_normaliser.fit(target)
        else:
            self.stain_normaliser = False
        
    def __getitem__(self, index):
        img                     = pil_loader(os.path.join(self.im_root, self.files[index] + self.ext))
        
        img                     = np.array(img)
        if self.stain_normaliser:
            img                 = self.stain_normaliser(img)
        
        if self.options.hed_decomp:
            img                 = skcolor.rgb2hed(img / 255.0).astype(np.float32)
            img                 = img[:,:,self.options.hed_channels]
            
        img                     = vtransforms.Compose(
                                    [vtransforms.ToTensor(), 
                                     vtransforms.Normalize(self.means, self.stds)]
                                  )(img)
        
        return img, self.files[index]
    
    def __len__(self):
        return len(self.files)

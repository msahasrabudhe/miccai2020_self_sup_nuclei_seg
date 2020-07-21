from train import *

from post import post as post_processing

import scipy.spatial.distance as spdist


import tqdm




def test_fn(sys_string=None):
    # ============================================
    #   Command line arguments. 
    parser                  = argparse.ArgumentParser()

    # Path the config file to use for this experiment. 
    parser.add_argument('--cfg', default='', type=str, help='Path to configuration file.')
    # Which epoch to use. 
    parser.add_argument('--epoch', default=-1, type=int, help='Use models from this epoch. -1 signifies use the latest model')
    # Path to data directory or file. 
    parser.add_argument('--dataroot', default='', type=str, help='Path to directory containing test images.')
    # File extension for test images. 
    parser.add_argument('--ext', default='.tif', type=str, help='File extension for test images.')
    # Whether to use a GPU and which GPU to use. A value of -1 indicates no CUDA.
    parser.add_argument('--gpu', default=-1, type=int, help='Which GPU to use. A value of -1 signifies no CUDA.')
    # Directory where to write output and save models/results/state_dicts. 
    parser.add_argument('--output_dir', default='output/', type=str, help='Path to directory which will store output.')

    # Whether to use arguments to main or from the command line. 
    if sys_string is not None:
        args                = parser.parse(sys_string.split(' '))
    else:
        args                = parser.parse_args()
    # ============================================

    assert args.gpu == -1 or args.gpu < torch.cuda.device_count(), \
            '--gpu must be either -1 or must specify a valid GPU ID (num_gpus={}). Got {}'.format(torch.cuda.device_count(), args.gpu)

    if args.gpu == -1:
        device              = 'cpu'
    else:
        device              = 'cuda:%d' %(args.gpu)


    options                 = load_yaml(args.cfg)
    fix_backward_compatibility(options)
   
    # Get the experiment name. This will be used to determine the output directory. 
    __, cfgname             = os.path.split(args.cfg)
    exp_name                = cfgname.replace('.yaml', '')

    # Determine output directory. 
    output_dir              = os.path.join(args.output_dir, exp_name)
    options.output_dir      = output_dir
    # Make sure the output directory exists. 
    assert os.path.exists(output_dir)


    # Print experiment configuration
    print('Experiment configuration')
    print('========================')
    print_config(options)
    print('========================')



    with open(os.path.join(output_dir, 'system_state.pkl'), 'rb') as fp:
        system_state        = pickle.load(fp)

    # Making output directory
    test_output_dir         = os.path.join(output_dir, 'test_output/')
    if not os.path.exists(test_output_dir):
        os.makedirs(test_output_dir)

    align_left('Initialising %s ...' %(options.model_arch))
    model                   = eval(options.model_arch)(options).to(device)
    write_okay()
  
    align_left('Loading saved models ...')
    nets_path               = os.path.join(output_dir, '%d'%(args.epoch), 'net.pth') if args.epoch != -1 else os.path.join(output_dir, 'net.pth')
    nets = load_state(nets_path)
    model.load_state_dict(nets['model'])
    write_okay()

    # Put model in eval mode
    model.eval();

    align_left('Building dataset ...')
    dataset                 = TestDataset(options, ext=args.ext, dataroot=args.dataroot, filenames=None)
    write_okay()


    align_left('Initialising dataloader ...')
    dataloader              = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4)
    write_okay()

    align_left('Defining list of transforms ...')
    transforms              = ['ID']
    if options.equivariance_scale:
        transforms.append('D2')
    if options.equivariance_aug:
        transforms += options.equivariance_aug 
    write_okay()

    for batch_idx, data_point in tqdm.tqdm(enumerate(loader, 0), ascii=True, ncols=100, desc='Computing attentions'):
        imgs                = data_point[0].to(device)
        files               = data_point[1] if batch_idx == 0 else files + data_point[1]
     
        att                 = None
        feats               = None
        for tr in TRANSFORMS_TO_USE:
            forward_fn      = AUG_TRANSFORMS_DICT[tr]['forward']
            backward_fn     = AUG_TRANSFORMS_DICT[tr]['backward']
            with torch.no_grad():
                out         = model(forward_fn(imgs), train=False, attention_only=True)
                
            att             = backward_fn(out['attention']) if att is None else torch.cat([att, backward_fn(out['attention'])], dim=1)
        att                 = att.mean(dim=1, keepdim=True)
        
        ims                 = imgs if batch_idx == 0 else torch.cat([ims, imgs], dim=0)
        segs                = att if batch_idx == 0 else torch.cat([segs, att], dim=0)

    # Post processing. 
    align_left('Post processing ...')
    posts                   = []
    for i in range(segs.shape[0]):
        seg                 = segs[i,0,:,:] > 0.5
        post                = post_processing(seg)
        posts.append(post)
    write_okay()


    align_left('Writing result ...')
    for f in files:
        out_file_name       = os.path.join(test_output_dir, f+'.tif')
        seg                 = posts[i]
        imageio.imsave(out_file_name, seg)
    write_okay() 

    print('[ DONE ]')     


if __name__ == '__main__':
    test_fn()
    

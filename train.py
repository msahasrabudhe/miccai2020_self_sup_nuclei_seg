from    datasets        import  *
from    models          import  *
from    utils           import  *
from    attr_dict       import  *

import  argparse
import  os
import  sys
from    IPython         import  embed
import  time
from    contextlib      import  ExitStack

from    sklearn         import  metrics

from    tensorboardX    import  SummaryWriter



def main(sys_string=None):
    """
    Main function. 
    Defines all training, and validation code. 
    
    Arguments
        sys_string          string          Command line arguments to pass to main. 
                                            Allows main to be executed in a Python console. 
                                            Default is None, which reads arguments from the 
                                            command line. 
    
    Return Value
        [None]
    """

    # ============================================
    #   Command line arguments. 
    parser                  = argparse.ArgumentParser()

    # Path the config file to use for this experiment. 
    parser.add_argument('--cfg', default='', type=str, help='Path to configuration file.')
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


    # Make sure that the configuration file exists. 
    assert os.path.exists(args.cfg), 'Specified cfg file {} does not exist!'.format(args.cfg)


    # Build an AttrDict from args.cfg which contains the experiment configuration. 
    # See attr_dict.py for a definition of AttrDict
    options                 = load_yaml(args.cfg)
    # Fix options for backward compatibility. This function allows old configuration files
    #   to be used while the codebase is updated. See utils.py for BKWD_CMPTBL_DICT
    #   and an explanation of how backward compatibility is ensured. 
    fix_backward_compatibility(options)


    # Print experiment configuration
    print('Experiment configuration')
    print('========================')
    print_config(options)
    print('========================')


    assert args.gpu == -1 or args.gpu < torch.cuda.device_count(), \
            '--gpu must be either -1 or must specify a valid GPU ID (num_gpus={}). Got {}'.format(torch.cuda.device_count(), args.gpu)

    if args.gpu == -1:
        device              = 'cpu'
    else:
        device              = 'cuda:%d' %(args.gpu)


    # Get the dataset name. 
    if options.dset_name not in ACCEPTED_DATASETS:
        raise ValueError('Unrecognised dataset. Must be one of ', ACCEPTED_DATASETS)

    # Dataset name determines which dataset to use. 
    if options.dset_name in [MONUSEGWSI]:
        ImageDataset        = MoNuSegWSIImageset
        EqualSampler        = MoNuSegWSIScaleEqualSampler

    elif options.dset_name in [MONUSEG]:
        ImageDataset        = MoNuSegImageset
        EqualSampler        = MoNuSegScaleEqualSampler


    # Get the experiment name. This will be used to determine the output directory. 
    __, cfgname             = os.path.split(args.cfg)
    exp_name                = cfgname.replace('.yaml', '')


    # Create sub-directories inside the output directory to record intermediate results. 
    imgs_to_record          = ['input', 'recon', 'foreground', 'background', \
                              'detection_map', 'attention', 'fg', 'bkg', \
                              'bk_aug_attn', 'at_0', 'attn_max', 'attn_mone']

    # Determine output directory. 
    output_dir              = os.path.join(args.output_dir, exp_name)
    options.output_dir      = output_dir

    # resuming specifies whether we are resuming a previously saved experiment. 
    resuming                = False
    # Check whether output_dir already exists. If yes, then there might be an aborted 
    #   experiment which will be continued. 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        os.makedirs(os.path.join(output_dir, 'best_model/'))
        os.makedirs(os.path.join(output_dir, 'images/'))
        for dname in imgs_to_record:
            os.makedirs(os.path.join(output_dir, 'images/', dname))
    else:
        # If the output directory already exists, check whether a previous training session is recorded. 
        # In case it is recorded, resume that session instead of starting from a fresh model.
        if os.path.exists(os.path.join(output_dir, 'system_state.pkl')):
            print('Found previous training instance in %s. Will attempt to resume using the previously saved state.' %(output_dir))
            options.load    = output_dir
            resuming        = True
        try:    os.makedirs(os.path.join(output_dir, 'images/'))
        except: pass

        for dname in imgs_to_record:
            try:    os.makedirs(os.path.join(output_dir, 'images/', dname))
            except: pass


    # ===============================================================
    #   Model definitions
    # ===============================================================

    # ===============================================================
    #   Define the attention net F and feature extractor for scale net G
    align_left('Initialising %s ...' %(options.model_arch))
    model                   = eval(options.model_arch)(options).to(device)
    write_okay()
    # If we wish to use a pretrained model, options.init_model specifies this. 
    # We load a previous state dict here. 
    if options.init_model != '':
        align_left('Loading trained model from %s ...' %(options.init_model))
        # Saved models are recorded as dictionaries. 
        prev_model          = load_state(options.init_model)['model']
        __state_dict        = model.state_dict()
        # Readjust the state dict so that only the part of the model
        #   that is stored in the state dict is loaded. 
        __pretrained_dict   = {}
        for kmodel, vmodel in __state_dict.items():
            if kmodel in prev_model.keys():
                __pretrained_dict[kmodel] = prev_model[kmodel]
            else:
                __pretrained_dict[kmodel] = vmodel
        model.load_state_dict(__pretrained_dict)
        write_okay()


    # ===============================================================
    #   Define a linear model which classifies scale from the feature
    #   vector generated by model. 
    align_left('Initialising %s ---' %(options.scale_arch))
    scale_net               = eval(options.scale_arch)(options).to(device)
    write_okay()
    # If we wish to use a pretrained model, options.init_model specifies this. 
    # We load a previous state dict here. 
    if options.init_model != '':
        align_left('Loading trained scale_net from %s ...' %(options.init_model))
        prev_model          = load_state(options.init_model)['scale_net']
        __state_dict        = scale_net.state_dict()
        # Readjust the state dict so that only the part of the scale_net
        #   that is stored in the state dict is loaded. 
        __pretrained_dict   = {}
        for kmodel, vmodel in __state_dict.items():
            if kmodel in prev_model.keys():
                __pretrained_dict[kmodel] = prev_model[kmodel]
            else:
                __pretrained_dict[kmodel] = vmodel
        scale_net.load_state_dict(__pretrained_dict)
        write_okay()


    # models_list specifies which models are to be trained. 
    models_list             = [scale_net]
    if not options.freeze_model:
        models_list.append(model)
    # Put in eval mode by default. 
    model.eval()


    # print models to be trained. 
    for model_ in models_list:
        print(model_)

    # =========================================================
    #   Create optimiser
    align_left('Creating optimiser [%s] ...' %(options.optimiser))
    # Group all parameters of models_list first. 
    params                  = group_params(models_list)
    if options.optimiser == 'sgd':
        optimiser           = optim.SGD(params, lr=options.lr, weight_decay=options.weight_decay, momentum=options.momentum)
    elif options.optimiser == 'adam':
        optimiser           = optim.Adam(params, lr=options.lr, weight_decay=options.weight_decay, betas=(options.momentum, 0.999))
    write_okay()

    # If a previous training session exists, load it. 
    if resuming:
        with open(os.path.join(output_dir, 'system_state.pkl'), 'rb') as fp:
            system_state    = pickle.load(fp)

        epoch               = system_state['epoch']
        iter_mark           = system_state['iter_mark']
        train_history       = system_state['train_history']
        val_history         = system_state['val_history']

        align_left('Loading previously saved model ...')
        assert os.path.exists(os.path.join(output_dir, 'net.pth')),       'No saved model found at {}!'.format(os.path.join(output_dir, '_net.pth'))
        assert os.path.exists(os.path.join(output_dir, 'optimiser.pth')), 'No saved optimiser found at {}!'.format(os.path.join(output_dir, 'optimiser.pth'))
        nets                = load_state(os.path.join(output_dir, 'net.pth'))
        optimisers          = load_state(os.path.join(output_dir, 'optimiser.pth'))

        model.load_state_dict(nets['model'])
        scale_net.load_state_dict(nets['scale_net'])
        optimiser.load_state_dict(optimisers)
        write_okay()
    else:
        epoch               = 0
        iter_mark           = 0
        train_history       = []
        val_history         = []


    # ===========================================================================================================================
    #   Tensorboard logging
    options.log_dir         = os.path.join('logs/', exp_name)
    writer                  = SummaryWriter(options.log_dir, purge_step=epoch-1)
        

    # Build loss scaling dictionary. 
    # if options.scale_<loss_name> is specified, this is the scaling
    #   parameter used to weight the loss. Otherwise, a value of 1 is used. 
    align_left('Building loss scaling dictionary ...')
    loss_scaling_dict       = {}
    for _loss in options.losses:
        if 'scale_'+_loss in options:
            loss_scaling_dict[_loss] = options['scale_'+_loss]
        else:
            loss_scaling_dict[_loss] = 1.0
    write_okay()


    # Make datasets.
    # Create transforms for train images. 
    align_left('Bulding datasets ...')
    train_transforms        = vtransforms.Compose([
#                                vtransforms.RandomCrop(random_crop_size, pad_if_needed=True, fill=0),
                                RandomRotation([0, 90, 180, 270]),
                                vtransforms.RandomHorizontalFlip(p=0.5),
                                vtransforms.RandomVerticalFlip(p=0.5),
                              ])
    train_dataset           = ImageDataset(options, 
                                        splits=['train'],
                                        img_transforms=train_transforms)

    val_dataset             = ImageDataset(options, 
                                        splits=['val'],
                                        img_transforms=None)
    write_okay()


    # =========================================================================
    #   Define a closure. 
    #   A closure determines how a data point is treated in the entire pipeline. 
    #   It takes in a batch of images sampled by the dataloader, 
    #   passed it through the architecture, computes losses and returns them. 
    align_left('Defining closure ...')
    def closure(data_point, train=False):
        # Images are the first element in data_point
        images              = data_point[0].to(device)
        # Targets (scale ground truth) are the second element. 
        targets             = data_point[1].to(device)
       
        # A dictionary which returns extra stuff (images etc.)
        extra               = {}

        # Pass the images through the attention network F and ResNet34. 
        out                 = model(images, train=train, return_h_feats=False)
        # features represents the encoding given by the ResNet34 before scale classification. 
        features            = out['fg_feats']
        # Pass the features through scale_net which is a linear model. 
        scores              = scale_net(features, train=train)

        # Record images. 
        extra['input']      = images
        extra['fg_feats']   = features
        extra['scores']     = scores
        extra['targets']    = targets
        if 'attention' in out:
            attention       = out['attention']
            extra['attention'] = out['attention']
        if 'at_0' in out:
            extra['at_0']   = out['at_0']
        if 'cl_logits' in out:
            cl_logits       = out['cl_logits']
            extra['cl_logits']   = out['cl_logits']


        # Equivariance constraints. Pass the batch through the attention network again, but this time
        #   transformed using a randomly chosen transformation t. options.equivariance_aug 
        #   specifies the set of transformations to use (\mathcal{T} in the paper). 
        if len(options.equivariance_aug) > 0:
            # Choose a transform at random.
            rand_aug            = random.choice(options.equivariance_aug)
            # Find the forward and backward functions for this transform. 
            forward_aug_fn      = AUG_TRANSFORMS_DICT[rand_aug]['forward']
            backward_aug_fn     = AUG_TRANSFORMS_DICT[rand_aug]['backward']
            # Transform the images. 
            with torch.no_grad():
                aug_img         = forward_aug_fn(images)
                # Compute the attention map on the transformed images. 
                aug_out         = model(aug_img, train=False, return_h_feats=False)
                aug_attention   = aug_out['attention']
                bk_aug_attn     = backward_aug_fn(aug_attention)
                extra['bk_aug_attn']  = bk_aug_attn

        # Equivariance to scale needs to be handled separately as only 20x and 40x patches can be downsampled. 
        if options.equivariance_scale:
            # Indices in batch representing 40x patches. 
            max_level_ids       = torch.LongTensor([x for x in torch.arange(targets.shape[0]) if targets[x].item() == len(options.levels)-1]).to(device)
            # Indices in batch representing 20x patches. 
            mone_level_ids      = torch.LongTensor([x for x in torch.arange(targets.shape[0]) if targets[x].item() == len(options.levels)-2]).to(device)

            # 'D2' is downsampling by 2. 
            forward_aug_fn      = AUG_TRANSFORMS_DICT['D2']['forward']
            backward_aug_fn     = AUG_TRANSFORMS_DICT['D2']['backward']

            # Transform the images. 
            with torch.no_grad():
                aug_max         = forward_aug_fn(images[max_level_ids,:,:,:])
                aug_mone        = forward_aug_fn(images[mone_level_ids,:,:,:])

                max_out         = model(aug_max, train=False, return_h_feats=False)
                mone_out        = model(aug_mone, train=False, return_h_feats=False)

                attn_max        = max_out['attention']
                attn_mone       = mone_out['attention']
                extra['attn_max']   = attn_max
                extra['attn_mone']  = attn_mone



        # This dictionary defines the losses.
        # options.losses determines which losses are to be used for training. 
        loss_definitions    = {
            'scale'             :   lambda : F.cross_entropy(scores, targets),
            'smooth_att'        :   lambda : l1_smoothness_loss(attention),
            'equiv_aug_mse'     :   lambda : F.mse_loss(attention, bk_aug_attn),
            'equiv_scale_mse'   :   lambda : 0.5 * (F.mse_loss(forward_aug_fn(attention[max_level_ids,:,:,:]), attn_max) + \
                                                    F.mse_loss(forward_aug_fn(attention[mone_level_ids,:,:,:]), attn_mone)),
        }
            
        # This dictionary defines predictions. 
        # options.predictions defines what predictions are to be made. 
        pred_definitions    = {
            'scale'         :   lambda : {
                                          'true':    targets.detach().cpu().numpy(), 
                                          'pred':    torch.argmax(scores, dim=1).detach().cpu().numpy(),
                                          'score':   scores.detach().cpu().numpy()
                                         },
        }
    
        # Compute all losses and weight them by their scaling hyperparameter. 
        losses              = [ loss_scaling_dict[loss_] * loss_definitions[loss_]() \
                                for loss_ in options.losses ]
        # Compute all predictions. 
        preds               = [ pred_definitions[loss_]() 
                                for loss_ in options.predictions ]

        # Return the result. 
        return losses, preds, extra

    write_okay()


    align_left('Defining train epoch ...')
    # ====================================================================
    #   Train epoch. 
    #   Define the training epoch here. This functions handles the use of
    #   the previously defined closure for a training epoch. 
    def train_epoch(loader):
        nonlocal iter_mark
        nonlocal train_history

        # Record all losses for this epoch. 
        losses_total        = [0 for l_ in options.losses]
        n_imgs              = 0

        # ETA. 
        start_time          = time.time()
        max_batches         = options.max_train_batches_per_epoch
        final_iter          = len(loader) if max_batches == -1 else max_batches


        # All predictions for this epoch. 
        all_preds           = {}

        extra               = {}

        # Iterate over the entire batch. 
        for batch_idx, data_point in enumerate(loader, 0):
            if batch_idx == final_iter:
                break

            # Reset gradients. 
            optimiser.zero_grad()

            batch_size      = data_point[0].size(0)
            n_imgs         += batch_size


            # Pass the data point through the closure in order to compute losses. 
            losses, preds, extra_  = closure(data_point, train=True)

            # Backward on losses. 
            sum(losses).backward()
            # Learn
            optimiser.step()

            # Record predictions for all images in the batch. 
            for m_, metric_ in enumerate(options.predictions, 0):
                all_preds[metric_]  = preds[m_] if metric_ not in all_preds \
                                      else { k: np.concatenate([all_preds[metric_][k],preds[m_][k]],axis=0) 
                                            for k in all_preds[metric_]}

            # Time taken. Helpful. 
            cur_time        = time.time()
            elapsed         = cur_time - start_time
            avg_per_it      = elapsed / (batch_idx + 1)
            remain          = final_iter - batch_idx - 1
            ETA             = remain * avg_per_it

            elapsed_m       = np.int(elapsed) // 60
            elapsed_s       = elapsed - 60 * elapsed_m
            ETA_m           = np.int(ETA) // 60
            ETA_s           = ETA - 60 * ETA_m

            # Update all losses for this epoch. 
            losses_total    = [L_ + l_*batch_size for L_,l_ in zip(losses_total, losses)]

            # Print iteration summary and log progress. 
            write_flush('TRAIN: Epoch %04d, batch %03d / %03d, iter %07d | ' %(epoch, batch_idx, final_iter-1, iter_mark))
            write_flush('Elapsed: %02dm%02ds .. ETA: %02dm%02ds | '%(elapsed_m, elapsed_s, ETA_m, ETA_s))
            write_flush('total: %.4f . ' %(sum(losses).item()))
            # Add to tensorboard logging. 
            writer.add_scalar('iter_total', sum(losses).item(), epoch)
            for li, loss_ in enumerate(options.losses, 0):
                write_flush('%s: %.4f . ' %(loss_, losses[li].item()))
                # Add to SummaryWriter
                writer.add_scalar('iter_'+loss_, losses[li].item(), epoch)

            # Print also the value of the threshold so far. 
            if hasattr(model, 'tau'):
                write_flush('tau: %.4f . ' %(model.tau.item()))
                writer.add_scalar('iter_tau', model.tau.item(), epoch)
            # Also print the sparsity. 
            if 'attention' in extra_:
                sparsity    = extra_['attention'].mean().item()
                write_flush('sparsity: %.4f . ' %(sparsity))
                writer.add_scalar('iter_sparsity', sparsity, epoch)
            write_flush('\n')

            # Increase iter_mark
            iter_mark      += 1
       
        # Divide the total loss by the number of images. 
        losses_total        = [L_ / n_imgs for L_ in losses_total]

        # Compute statistics and store history. 
        train_history_data_point = {}
        # Add the total loss. 
        train_history_data_point['total'] = sum(losses_total).item()
        # Add to SummaryWriter
        writer.add_scalar('train_total', sum(losses_total).item(), epoch)
        for li, loss_ in enumerate(options.losses):
            train_history_data_point[loss_] = losses_total[li].item()
            # Add to SummaryWriter
            writer.add_scalar('train_'+loss_, losses_total[li].item(), epoch)
            
        for metric_ in options.predictions:
            y_true      = all_preds[metric_]['true']
            y_pred      = all_preds[metric_]['pred']
            acc_        = metrics.accuracy_score(y_true, y_pred)
            bal_acc_    = metrics.balanced_accuracy_score(y_true, y_pred)

            train_history_data_point['true_'+metric_] = y_true
            train_history_data_point['pred_'+metric_] = y_pred
            train_history_data_point['acc_'+metric_] = acc_
            train_history_data_point['bal_acc_'+metric_] = bal_acc_
            # Add to SummaryWriter
            writer.add_scalar('acc_'+metric_, acc_, epoch)
            writer.add_scalar('bal_acc_'+metric_, bal_acc_, epoch)

        write_flush('Train epoch %04d summary\n=======================\n' %(epoch))
        for key in ['total'] + options.losses + [item for m in options.predictions for item in [m, 'acc_'+m, 'bal_acc_'+m]]:
            write_flush('\t%20s: %.4f\n' %(key, train_history_data_point[key]))
       
        train_history.append(train_history_data_point)

        for k in extra_:
            extra[k]    = extra_[k]

        return losses_total, extra

    write_okay()

    align_left('Defining test epoch ...')
    # ================================================================
    #   Test epoch. 
    #   Defines how the closure is used for testing. 
    def test_epoch(loader):
        nonlocal val_history

        losses_total        = [0 for l_ in options.losses]
        n_imgs              = 0

        start_time          = time.time()
        max_batches         = options.max_val_batches_per_epoch
        final_iter          = len(loader) if max_batches == -1 else max_batches

        all_preds           = {}
        extra               = {}

        for batch_idx, data_point in enumerate(loader, 0):
            if batch_idx == final_iter:
                break

            batch_size      = data_point[0].size(0)
            n_imgs         += batch_size

            # Fix data point, because it looks like [batch_size, NT, C, H, W]
            # where NT is the number of crops. 
            if data_point[0].dim() == 5:
                _, n_transform, C, H, W = data_point[0].size()
                data_point[0]   = data_point[0].view(-1, C, H, W)
                data_point[1]   = data_point[1].unsqueeze(1).repeat(1,n_transform).view(-1)
            else:
                n_transform    = -1

            losses, preds, extra_  = closure(data_point, train=False)

            if n_transform != -1:
                for m_, metric_ in enumerate(options.predictions, 0):
                    new_score = preds[m_]['score'].reshape([batch_size, n_transform, -1]).mean(axis=1)
                    new_target  = preds[m_]['true'].reshape(batch_size, n_transform)[:,0]
                    preds[m_]['true'] = new_target
                    preds[m_]['pred'] = new_score.argmax(axis=1)
                    preds[m_]['score'] = new_score
            
            for m_,  metric_ in enumerate(options.predictions, 0):
                all_preds[metric_]  = preds[m_] if metric_ not in all_preds \
                                      else { k: np.concatenate([all_preds[metric_][k],preds[m_][k]],axis=0) 
                                            for k in all_preds[metric_]}

            # Time taken. Helpful. 
            cur_time        = time.time()
            elapsed         = cur_time - start_time
            avg_per_it      = elapsed / (batch_idx + 1)
            remain          = final_iter - batch_idx - 1
            ETA             = remain * avg_per_it

            elapsed_m       = np.int(elapsed) // 60
            elapsed_s       = elapsed - 60 * elapsed_m
            ETA_m           = np.int(ETA) // 60
            ETA_s           = ETA - 60 * ETA_m

            losses_total    = [L_ + l_*batch_size for L_,l_ in zip(losses_total, losses)]

            write_flush('TEST : Epoch %04d, batch %03d / %03d, iter %07d | ' %(epoch, batch_idx, final_iter-1, iter_mark))
            write_flush('Elapsed: %02dm%02ds . ETA: %02dm%02ds | '%(elapsed_m, elapsed_s, ETA_m, ETA_s))
            write_flush('total: %.4f . ' %(sum(losses).item()))
            # Add to tensorboard logging. 
            writer.add_scalar('iter_total', sum(losses).item(), epoch)
            for li, loss_ in enumerate(options.losses, 0):
                write_flush('%s: %.4f . ' %(loss_, losses[li].item()))
                # Add to SummaryWriter
                writer.add_scalar('iter_'+loss_, losses[li].item(), epoch)

            if hasattr(model, 'tau'):
                write_flush('tau: %.4f . ' %(model.tau.item()))
            if 'attention' in extra_:
                sparsity    = extra_['attention'].mean().item()
                write_flush('sparsity: %.4f . ' %(sparsity))
            write_flush('\n')
        
        losses_total        = [L_ / n_imgs for L_ in losses_total]

        # Compute statistics and store history. 
        val_history_data_point  = {}
        # Add the total loss. 
        val_history_data_point['total'] = sum(losses_total).item()
        # Add to SummaryWriter
        writer.add_scalar('train_total', sum(losses_total).item(), epoch)
        for li, loss_ in enumerate(options.losses):
            val_history_data_point[loss_] = losses_total[li].item()
            # Add to SummaryWriter
            writer.add_scalar('train_'+loss_, losses_total[li].item(), epoch)
            
        for metric_ in options.predictions:
            y_true      = all_preds[metric_]['true']
            y_pred      = all_preds[metric_]['pred']
            acc_        = metrics.accuracy_score(y_true, y_pred)
            bal_acc_    = metrics.balanced_accuracy_score(y_true, y_pred)

            val_history_data_point['true_'+metric_] = y_true
            val_history_data_point['pred_'+metric_] = y_pred
            val_history_data_point['acc_'+metric_] = acc_
            val_history_data_point['bal_acc_'+metric_] = bal_acc_
            # Add to SummaryWriter
            writer.add_scalar('acc_'+metric_, acc_, epoch)
            writer.add_scalar('bal_acc_'+metric_, bal_acc_, epoch)

        write_flush('Validation epoch %04d summary\n=======================\n' %(epoch))
        for key in ['total'] + options.losses + [item for m in options.predictions for item in [m, 'acc_'+m, 'bal_acc_'+m]]:
            write_flush('\t%20s: %.4f\n' %(key, val_history_data_point[key]))
       
        val_history.append(val_history_data_point)

        for k in extra_:
            extra[k]    = extra_[k]

        return losses_total, extra


    write_okay()

    # =========================================================================
    #   Learning.
    while epoch < options.n_epochs:

        # Decay learning rate. 
        if epoch in options.lr_decay_every:
            align_left('Decaying learning rates ...')
            for opt in [optimiser]:
                for pg in opt.param_groups:
                    pg['lr']       *= options.lr_decay
            write_okay()


        # =====================================================================
        #   Training phase. 
        write_flush('Training ...\n')
        for model_ in models_list:
            model_.train()

        # Sampler for sampling training images. 
        # Samples images of all classes uniformly. 
        equal_sampler           = EqualSampler(train_dataset)

        # Create dataloaders
        train_dataloader        = torch.utils.data.DataLoader(train_dataset, 
                                                            batch_size=options.batch_size, 
                                                            sampler=equal_sampler, 
                                                            shuffle=False,
                                                            num_workers=options.workers)

        losses, _               = train_epoch(train_dataloader)
        # =====================================================================


        # =====================================================================
        #   Validation phase. 
        write_flush('Validating ...\n')
        for model_ in models_list:
            model_.eval()

        # Sampler for sampling validation images. 
        # Samples images of all classes uniformly. 
        equal_sampler           = EqualSampler(val_dataset)

        val_dataloader          = torch.utils.data.DataLoader(val_dataset, 
                                                            batch_size=options.batch_size, 
                                                            shuffle=False,
                                                            num_workers=options.workers,
                                                            sampler=equal_sampler)
        with torch.no_grad():
            losses, extra       = test_epoch(val_dataloader)
        # =====================================================================

        system_state    = {
            'train_history'         : train_history,
            'val_history'           : val_history, 
            'epoch'                 : epoch + 1,
            'iter_mark'             : iter_mark,
        }
        with open(os.path.join(output_dir, 'system_state.pkl'), 'wb') as fp:
            pickle.dump(system_state, fp)
 

        # Save models. 
        align_left('Saving models ...')
        model_dict              = {
            'model'             :   model.state_dict(),
            'scale_net'         :   scale_net.state_dict(),
        }
        torch.save(model_dict, os.path.join(output_dir, 'net.pth'))
        torch.save(optimiser.state_dict(), os.path.join(output_dir, 'optimiser.pth'))
        write_okay()

        if (isinstance(options.checkpoint_special, list) and (epoch + 1) in options.checkpoint_special) or\
           (isinstance(options.checkpoint_special, int) and (epoch + 1)%options.checkpoint_special == 0):
            align_left('Checkpointing for epoch %d ...' %(epoch+1))
            try:
                os.makedirs(os.path.join(output_dir, '%d'%(epoch)))
            except:
                pass

            torch.save(model_dict, os.path.join(output_dir, '%d'%(epoch), 'net.pth'))
            torch.save(optimiser.state_dict(), os.path.join(output_dir, '%d'%(epoch), 'optimiser.pth'))
            write_okay()

        # Saving images. 
        # Add input image. 
        to_record               = ['input']
        # Add extra images. 
        to_record              += [ek for ek in ['attention', 'fg', 'bkg', 'recon', 'bk_aug_attn', 'at_0', 'attn_max', 'attn_mone'] if ek in extra]

        for k in to_record:
            if k == 'input' and extra[k].size(1) == 2:
                extra[k]        = extra[k][:,[0],:,:]
            G                   = vutils.make_grid(extra[k], nrow=8, normalize=True, padding=2)
            imageio.imsave(os.path.join(output_dir, 'images/', k, '%08d.png' %(epoch)), np.uint8(np.floor(G.permute(1,2,0).detach().cpu().numpy() * 255)))
            writer.add_image(k, G, epoch)
        

        print('--> Epoch %d model, optimiser, and system state saved to %s.' %(epoch, output_dir))

        epoch          += 1

    return

if __name__ == '__main__':
    main()

import  argparse, os, inspect


# arguments for model  
def parse_args():
    parser = argparse.ArgumentParser(description="Args for training parameters")
   
    parser.add_argument('--epoch', type=int, default=10, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=64, help='The size of batch')
    parser.add_argument('--ind', type=int,default=1, help='Experiment index') 
    
    ## directory argument 
    parser.add_argument('--save_dir', type=str, default=str(inspect.getfile(inspect.currentframe())[:-3]),
                        help='Directory name to save the model')
    parser.add_argument('--result_dir', type=str, default='.\\results_v3\\', help='Directory name to save the generated images')
    parser.add_argument('--result_dir_eeg', type=str, default='EEG\\', help='Directory name to save the generated EEG images')
    parser.add_argument('--result_dir_eog', type=str, default='EOG\\', help='Directory name to save the generated EOG images')
    
    parser.add_argument('--log_dir', type=str, default='logs_v3', help='Directory name to save training logs')
    parser.add_argument('--root_dir', type=str, default='.\\data\\sleep_edfx', help='Directory name to save training logs')
    parser.add_argument('--eeg_path', type=str, default='EEG_train', help='EEG Directory name')
    parser.add_argument('--eog_path', type=str, default='EOG_train', help='EOG Directory name')
    
    # hyperparameter argument 
    parser.add_argument('--lrG', type=float, default=0.0002) 
    parser.add_argument('--mytemp', type=float, default=0.5)
    parser.add_argument('--klwt', type=float, default=10.0)
    parser.add_argument('--segment', type=float, default = 6)
    parser.add_argument('--lambda_g', type=float, default=0.5)
    parser.add_argument('--lambda_l', type=float, default=0.5)

    parser.add_argument('--lrD', type=float, default=0.0002)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    
    # mode argument 
    parser.add_argument('--gpu_mode', type=bool, default=True)
    return argument_chkr(parser.parse_args())


def argument_chkr(args):
    """verify arguments"""
    
    #save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs('{}\\saved_models'.format(args.save_dir))
        os.makedirs('{}\\saved_models\\netG'.format(args.save_dir))
        os.makedirs('{}\\saved_models\\netD'.format(args.save_dir))
        os.makedirs('{}\\saved_models\\netFE'.format(args.save_dir))
        os.makedirs('{}\\saved_models\\netQ'.format(args.save_dir))

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    return args

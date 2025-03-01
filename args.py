import argparse
import os
import random


def check_dirs(dirs):
    dirs = [dirs] if type(dirs) not in [list, tuple] else dirs
    for d in dirs:
        try:
            os.makedirs(d)
        except OSError:
            pass
    return


def get_args():
    parser = argparse.ArgumentParser()

    ## Common Parameters ##
    parser.add_argument('-D','--datasets', type=str, nargs='+', required=True, help='M/MM/SC/PC (MNIST/MNIST-M/Synthetic Can/Physical Can) ')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--imsize', type=int, help='the height of the input image')
    parser.add_argument('--iter', type=int, default=10000000, help='total training iterations')
    parser.add_argument('--manualSeed', type=int, default=5688)
    parser.add_argument('--ex', help='Experiment name')
    parser.add_argument('--logfile', type=str)
    parser.add_argument('--tensor_freq', type=int, help='frequency of showing results on tensorboard during training.')
    parser.add_argument('--net_freq', type=int, help='frequency of network saving during training.')
    parser.add_argument('--CADT', type=bool, default=True)
    parser.add_argument('--resume_checkpoint', type=bool, default=False, help="Set true to continue training from where you left off or generate data from checkpoint else False")
    parser.add_argument('--load_step', type=int, help="iteration of trained networks")
    parser.add_argument('--gen_data', default=True, type=bool, help="sets networks to eval mode for generating data")
    parser.add_argument('--save_step', default=1, type=int, help="the step where images start being saved")
    parser.add_argument('--num_imgs', default=2500, type=int, help="number of generated images saved")

    ## Optimizers Parameters ##
    parser.add_argument('--lr_cas', type=float, default=0.001)
    parser.add_argument('--lr_decay_rate', type=float, default=0.95)
    parser.add_argument('--lr_decay_step', type=int, default=600)#600 good for cans #6000 is good for MNIST
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--weight_decay_cas', type=float, default=1e-5)

    args = parser.parse_args()
    check_dirs(['checkpoint/' + args.ex])
    args.logfile = './checkpoint/' + args.ex + '/' + args.ex + '.log'
    if args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    if args.batch is None:
        args.batch = 1#2 for Cans, 32 for MNIST
    if args.imsize is None:
        args.imsize = 256#256 for Cans, 64 for MNIST
    if args.tensor_freq is None:
        args.tensor_freq = 1000
    if args.net_freq is None:
        args.net_freq = 500
    return args

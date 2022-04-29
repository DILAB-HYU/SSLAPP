import argparse, os, inspect

from torch.backends import cudnn
from model import SSLLAP
from args import parse_args


if __name__ == '__main__':
   
    args = parse_args()
    if args is None:
        exit()
    print("Model arguments: " ,args)

    if args.gpu_mode: 
        cudnn.benchmark = True

    print("Model save at:", inspect.getfile(inspect.currentframe())[:-3])

    model = SSLLAP(args)

    model.train()
    
    print("=============Training Finish=============")


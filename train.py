from __future__ import print_function
from args import get_args
from trainer import Trainer

if __name__ == '__main__':
    opt = get_args()
    trainer = Trainer(opt)
    trainer.train()

#python train.py -T clf -D M MM --ex M2MM
#python train.py -D SC PC --imsize 256 --save_step 100 --num_imgs 20 --tensor_freq 100 --ex SC2PC
#tensorboard --logdir tensorboard --bind_all
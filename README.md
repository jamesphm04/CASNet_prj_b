# [SMC 2024] Sim-to-Real Domain Adaptation for Deformation Classification.

## Requirements
```
torch 2.3.1
numpy 1.26.3
torchvision 0.18.1
matplotlib 3.9.0
pillow 10.2.0
scikit-learn 1.5.0
tensorboardX 2.6.2.2
prettytable 3.10.0
```
## Data Preparation
Download [MNIST-M](https://github.com/fungtion/DANN), [Physical Can Dataset](https://drive.google.com/drive/folders/19KR56Hvpdkcomvz7Y-ff3Mr0PLp9So1P)

Create Synthetic Can Dataset by following my other github project [Synthetic_Deformation_Data_Generation](https://github.com/JoelESol/Synthetic_Deformation_Data_Generation) and paper [Visual Deformation Detection Using Soft Material Simulation for Pre-training of Condition Assessment Models](http://arxiv.org/abs/2405.14877)
## Folder Structure of Datasets
```
├── data
      ├── MNIST
      ├── mnist_m
            ├── mnist_m_train
                      ├── *.png
            ├── mnist_m_test
                      ├── *.png
            ├── mnist_m_train_labels.txt
            ├── mnist_m_test_labels.txt
      ├── physical_can
            ├── deform_images
                    ├── *.png
            ├── truth_images
                    ├── *.png
            ├── image_data.txt
      ├── Synthetic_Can_Data
            ├── train
                    ├── *.png
            ├── train.txt
      ├── converted
            ├── PC2SC
                    ├── *.png
            ├── SC2PC
                    ├── *.png
            ├── PC2SC_converted_images.png
            ├── SC2PC_converted_images.png
```
## Train
Here are some example commands see args.py for more details
```
python train.py -D SC PC --imsize 256 --eval_freq 100 --tensor_freq 50 --ex SC2PC
python train.py -D SC PC --imsize 256 --resume_checkpoint True --load_step 1000 --eval_freq 100 --tensor_freq 50 --ex SC2PC
python train.py -D SC PC --imsize 256 --resume_checkpoint True --load_step 1000 --gen_data True --ex SC2PC
```
## Tensorboard
You can see all the results of each experiment on tensorboard.
```
tensorboard --logdir tensorboard --bind_all
```

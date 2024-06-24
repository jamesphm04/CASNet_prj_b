import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
from skimage.transform import resize
from sklearn.decomposition import PCA
import torchvision.transforms as trans
from torch.utils.data import DataLoader
from dataloader import *

imsize=256
# Define the synthetic dataset
synth_train_dataset = Synth_Dataset(root='./data/Synthetic_Can_Data', train=True,
                                    transform=trans.Compose([
                                        trans.Resize(imsize),
                                        trans.ToTensor(),
                                        trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))

# Define the physical dataset
pys_train_dataset = Phys_Dataset(root='./data/physical_can', train=True,
                                 transform=trans.Compose([
                                     trans.Resize(imsize),
                                     trans.ToTensor(),
                                     trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]))
genSC_train = Synth_Dataset(root='./data/SC_gen_data', train=True,
                              transform=trans.Compose([
                                  trans.Resize(imsize),
                                  trans.ToTensor(),
                                  trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                              ]))

print("loading")
data_synth = np.array([synth_train_dataset[i][0].numpy().flatten() for i in range(len(synth_train_dataset))])
data_phys = np.array([pys_train_dataset[i][0].numpy().flatten() for i in range(len(pys_train_dataset))])
data_genSC = np.array([genSC_train[i][0].numpy().flatten() for i in range(len(genSC_train))])


data = np.concatenate((data_synth, data_phys, data_genSC), axis=0)
labels = np.concatenate((np.zeros(len(synth_train_dataset)), np.ones(len(pys_train_dataset)), np.full(len(genSC_train), 2)), axis=0)
print("PCA")
# Perform PCA dimensionality reduction on the CPU
pca = PCA(n_components=2, random_state=42)
pca_data = pca.fit_transform(data)

# Plot the PCA visualization
plt.figure(figsize=(10, 8))
plt.scatter(pca_data[:len(data_synth), 0], pca_data[:len(data_synth), 1], c='r', alpha=0.5, label='Synthetic Black')
plt.scatter(pca_data[len(data_synth):len(data_synth) + len(data_genSC), 0], pca_data[len(data_synth):len(data_synth) + len(data_genSC), 1], c='m', alpha=0.5, label='Generated')
plt.scatter(pca_data[len(data_synth) + len(data_genSC):, 0], pca_data[len(data_synth) + len(data_genSC):, 1], c='b', alpha=0.5, label='Physical')


plt.title('PCA Visualization of Synthetic, Generated and Physical Datasets')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.legend()
plt.show()
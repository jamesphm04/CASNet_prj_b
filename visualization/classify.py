from dataloader import *
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as trans
from torch.utils.data import ConcatDataset
import models.CASNet as model
from sklearn.metrics import confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

imsize = 512

synth_train_dataset = Synth_Dataset(root='./data/Synthetic_Can_Data', train=True,
                                    transform=transforms.Compose([
                                        trans.Resize(imsize),
                                        trans.ToTensor(),
                                        trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    ]))
synth_test_dataset = Synth_Dataset(root='./data/Synthetic_Can_Data', train=False,
                                   transform=transforms.Compose([
                                       trans.Resize(imsize),
                                       trans.ToTensor(),
                                       trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                   ]))

phys_train_dataset = Phys_Dataset(root='./data/physical_can', train=True,
                                 transform=transforms.Compose([
                                     trans.Resize(imsize),
                                     trans.ToTensor(),
                                     trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]))
phys_test_dataset = Phys_Dataset(root='./data/physical_can', train=False,
                                transform=transforms.Compose([
                                    trans.Resize(imsize),
                                    trans.ToTensor(),
                                    trans.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))

# Create dataloaders
# Synthetic only classification
train_dataset = ConcatDataset([synth_train_dataset, synth_test_dataset])
test_dataset = ConcatDataset([phys_train_dataset, phys_test_dataset])

# Sim-to-Real Classification
#train_dataset = ConcatDataset([synth_train_dataset, synth_test_dataset])
#test_dataset = ConcatDataset([gen_train_PC, gen_test_PC])

batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize the model, loss function, and optimizer
net = model.VGG16()
net.cuda()
net.freeze_layers()

criterion = nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.005)

# Training loop
num_epochs = 25
for epoch in range(num_epochs):
    net.train()
    running_loss = 0.0
    for images, labels in train_dataloader:
        images, labels = images.to(device), labels.to(device)
        labels = labels.unsqueeze(1)
        labels = labels.float()
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_dataset)}")

    # Evaluate the model
    net.eval()
    total = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for images, labels in test_dataloader:
            images, labels = images.to(device), labels.to(device)
            labels = labels.unsqueeze(1)
            labels = labels.float()
            outputs = net(images)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(true_labels, predictions)

    # Extract TP, TN, FP, FN from the confusion matrix
    TN = conf_matrix[1, 1]
    TP = conf_matrix[0, 0]
    FN = conf_matrix[0, 1]
    FP = conf_matrix[1, 0]
    Acc = ((TP + TN) / (TP + TN + FP + FN)) * 100
    def_ACC = (TN / (FP + TN)) * 100
    non_ACC = (TP / (FN + TP)) * 100

    print(f'TP: {TP}')
    print(f'TN: {TN}')
    print(f'FP: {FP}')
    print(f'FN: {FN}')
    print(f'ACC: {Acc}')
    print(f'Deform ACC {def_ACC}')
    print(f'Non-def ACC {non_ACC}')

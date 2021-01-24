from models.LSTM import LSTM
from dataset.DEAP import DEAP
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
deap_dataset = DEAP(subject=10, num_segments=12)

train_indices, val_indices = deap_dataset.train_valid_split(split_ratio=0.2)
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(dataset=deap_dataset, batch_size=8, sampler=train_sampler)
val_dataloader = DataLoader(dataset=deap_dataset, batch_size=8, sampler=val_sampler)

emotion_classifier = LSTM().to(device)
mse_loss = nn.MSELoss()
bce_loss = nn.BCELoss()
optimizer = optim.Adam(emotion_classifier.parameters(), lr=0.005)

train_loss = []
val_loss = []
val_acc_all = []

num_epochs = 30

for epoch in range(num_epochs):
    epoch_loss = []
    print("Starting epoch: ", epoch)
    for i, batch in enumerate(train_dataloader):
        # Get the data
        # eeg_data = batch['eeg']
        face_data = batch['face'].to(device)
        y = batch['label_arousal'].to(device)  # Map to 0 or 1
        optimizer.zero_grad()  #
        preds = emotion_classifier(face_data)  # Forward pass
        loss = mse_loss(preds.squeeze(), y)  # Loss
        loss.backward()  ##
        optimizer.step()  ###
        epoch_loss.append(loss.item())

    train_loss.append(np.mean(epoch_loss))

    with torch.no_grad():
        epoch_val_loss = []
        val_acc_running = []
        for i, batch in enumerate(val_dataloader):
            face_data = batch['face'].to(device)
            y = (batch['label_arousal']).to(device)
            preds = emotion_classifier(face_data)
            epoch_val_loss.append(mse_loss(preds.squeeze(), y))
            preds = preds >= 5
            val_acc = float((preds.squeeze() == (y>5)).sum()/len(y))
            val_acc_running.append(val_acc)
        val_acc_all.append(np.mean(val_acc_running))
        val_loss.append(np.mean(epoch_val_loss))
        print("Validation accuracy: ", np.mean(val_acc_running))

plt.plot(np.arange(1, num_epochs+1), train_loss, label="Training")
plt.plot(np.arange(1, num_epochs+1), val_loss, label="Validation")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss.png")
plt.show()


plt.plot(np.arange(1, num_epochs+1), val_acc_all, label="Val accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("accuracy.png")
plt.show()

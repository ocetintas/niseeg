from models.LSTM import LSTM
from dataset.DEAP import DEAP
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn as nn
import torch.optim as optim
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
deap_dataset = DEAP(subject=12, num_segments=12)

train_indices, val_indices = deap_dataset.train_valid_split(split_ratio=0.2)
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_dataloader = DataLoader(dataset=deap_dataset, batch_size=8, sampler=train_sampler)
val_dataloader = DataLoader(dataset=deap_dataset, batch_size=8, sampler=train_sampler)

emotion_classifier = LSTM()
mse_loss = nn.MSELoss()
optimizer = optim.Adam(emotion_classifier.parameters(), lr=0.001)

train_loss = []
vall_acc = []

for epoch in range(30):
    print("Starting epoch: ", epoch)
    for i, batch in enumerate(train_dataloader):
        # Get the data
        # eeg_data = batch['eeg']
        face_data = batch['face']
        y = batch['label_arousal']  # Map between 0 and 1
        optimizer.zero_grad()  #
        preds = emotion_classifier(face_data)  # Forward pass
        loss = mse_loss(preds.squeeze(), y)  # Loss
        loss.backward()  ##
        optimizer.step()  ###
        if i%10 == 0:
            print(loss.item())

    val_acc_running = []
    for i, batch in enumerate(val_dataloader):
        face_data = batch['face']
        y = (batch['label_arousal']) >= 5
        preds = emotion_classifier(face_data)
        print(preds, y)
        preds = preds >= 5
        val_acc = (preds.squeeze() == y).sum()/len(y)
        val_acc_running.append(val_acc)
    print("Validation accuracy: ", np.mean(val_acc_running))
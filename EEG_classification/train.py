import torch.optim as optim
from torch.utils.data import DataLoader
from classifiers import EmotionLSTM
from dataset_eeg import EmotionEEG
## For LSTM

def label_to_class(labels, state, num_class):
    if state=='arousal':
        l=labels[:,0]
    elif state=='valence':
        l=labels[:,1]
    if num_class=2:
        cl=l>=0.5
        ## TODO: implment 3 classes case

    return cl.long()

def train(args):
    
    
    dataset=EmotionEEG(args.datapath)
    dataloader=DataLoader(dataset=dataset, batch_size = args.batch_size, shuffle=True)
    emotion=EmotionLSTM(args.num_class)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(emotion.parameters())

    for epoch in range(args.epochs):
        for i, batch in enumerate(dataloader):
            eeg=batch['eeg'].transpose(1,-1)
            cl=label_to_class(batch['label'], args.state, args.num_class)
            optimizer.zero_grad()
            pred = emotion(eeg).squeeze()
            loss = criterion(pred, cl)
            loss.backward()
            optimizer.step()
            ##TODO - Implement checkpoints, logs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arser.add_argument('datapath', type=str, help='dir of files to read data from')
    parser.add_argument('--batch-size', type=int, help='batch size of data loader')
    parser.add_argument('--num-class', type=int, help='2 classes or 3 classes problem')
    parser.add_argument('state', type=str), help='state to classify')
    args = parser.parse_args()
    train(args)

import torch.optim as optim
from torch.utils.data import DataLoader
from classifiers import EmotionLSTM
from dataset_eeg import EmotionEEG
import time
## For LSTM

def label_to_class(labels, state, num_class):
    if state=='arousal':
        l=labels[:,0]
    elif state=='valence':
        l=labels[:,1]
    if num_class=2:
        cl=l>=0.5
        ## TODO: implment 3 classes case

    return cl.float()

def train(args):
    
    
    dataset=EmotionEEG(args.datapath)
    train_indices, val_indices = dataset.train_valid_split(split_ratio=args.split_ratio, seed=args.seed)
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    train_dataloader=DataLoader(dataset=dataset, args.batch_size =64, sampler=train_sampler)
    valid_dataloader=DataLoader(dataset=dataset, args.batch_size =64, sampler=train_sampler)
    emotion=EmotionLSTM(args.num_class)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(emotion.parameters())

    train_losses=[]
    val_accs =[]
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    for epoch in range(30):
        for i, batch in enumerate(train_dataloader):
            eeg=batch['eeg'].transpose(1,-1)
            cl=label_to_class(batch['label'][:,0], 'arousal')
            optimizer.zero_grad()
            pred = emotion(eeg).squeeze().mean(dim=1)
            loss = criterion(pred, cl)
            loss.backward()
            optimizer.step()
            if i%5==1:
                train_losses.append(loss)
                print(f'Train loss: {loss}')
        val_accs_running=[]
        for i, batch in enumerate(valid_dataloader):
            pred = emotion(eeg).squeeze().mean(dim=1) > 0.5
            cl=label_to_class(batch['label'][:,0], 'arousal') ==1
            acc=(pred == cl).sum()/len(cl)
            val_accs_running.append(acc)
            
        val_accs.append(np.mean(val_accs_running))
        print(f'Valid acc: {np.mean(val_accs_running)}')

    torch.save(emotion.state_dict(), args.checkpoint+'/eeglstm_'+ timestr +'.pt')

    return train_losses, val_accs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    arser.add_argument('datapath', type=str, help='dir of files to read data from')
    parser.add_argument('--batch-size', type=int, help='batch size of data loader')
    parser.add_argument('--num-class', type=int, help='2 classes or 3 classes problem')
    parser.add_argument('state', type=str), help='state to classify')
    parser.add_argument('seed', type=int, help='train, valid split seed')
    parser.add_argument('--split-ratio', type=float, help='train, valid split ratio')
    parser.add_argument('checkpoint', type=str, hep='dir to save model')
    args = parser.parse_args()
    train(args)

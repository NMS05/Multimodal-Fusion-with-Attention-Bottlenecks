import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloader.av_data import AV_Dataset
from models.visual_model import AVmodel

def parse_options():
    parser = argparse.ArgumentParser(description="Multimodal Bottleneck Attention")

    ##### TRAINING DYNAMICS
    parser.add_argument('--gpu_id', type=str, default="cuda:0", help='the gpu id')
    parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
    parser.add_argument('--batch_size', type=int, default=8, help='batchsize')
    parser.add_argument('--num_epochs', type=int, default=15, help='total training epochs')
    parser.add_argument('--seed', type=int, default=1111, help='random seed')

    # Adapter and Latent Params
    parser.add_argument('--adapter_dim', type=int, default=8, help='dimension of the low rank')
    parser.add_argument('--num_latent', type=int, default=4, help='number of latent tokens')
    parser.add_argument('--num_classes', type=int, default=28, help='number of latent tokens')

    ##### DATA
    parser.add_argument('--audio_dir', type=str, default='audio_files', help='dir of audio files')
    parser.add_argument('--visual_dir', type=str, default='rgb_frames/', help='dir of rgb frames')

    opts = parser.parse_args()
    torch.manual_seed(opts.seed)
    opts.device = torch.device(opts.gpu_id)
    return opts

############################################################################################################################################################################################################
############################################################################################################################################################################################################

def train_one_epoch(train_data_loader,model,optimizer,loss_fn, device):
    
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0
    
    model.train()

    ###Iterating over data loader
    for spec, imgs, labels in train_data_loader:
        
        #Loading data and labels to device
        spec = spec.to(device)
        imgs = imgs.to(device)
        labels = labels.to(device)

        #Reseting Gradients
        optimizer.zero_grad()

        #Forward
        preds = model(spec, imgs)

        #Calculating Loss
        _loss = loss_fn(preds, labels)
        epoch_loss.append(_loss.item())
        
        #Backward
        _loss.backward()
        optimizer.step()

        sum_correct_pred += (torch.argmax(preds,dim=1) == labels).sum().item()
        total_samples += len(labels)

    acc = round(sum_correct_pred/total_samples,5)*100
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, acc

def val_one_epoch(val_data_loader, model,loss_fn, device):
    
    ### Local Parameters
    epoch_loss = []
    sum_correct_pred = 0
    total_samples = 0

    model.eval()

    with torch.no_grad():
      for spec, imgs, labels in val_data_loader:

        #Loading data and labels to device
        spec = spec.to(device)
        imgs = imgs.to(device)
        labels = labels.to(device)

        #Forward
        preds = model(spec, imgs)
        
        #Calculating Loss
        _loss = loss_fn(preds, labels)
        epoch_loss.append(_loss.item())
        
        # print(torch.argmax(preds,dim=1),labels)
        sum_correct_pred += (torch.argmax(preds,dim=1) == labels).sum().item()
        total_samples += len(labels)

    acc = round(sum_correct_pred/total_samples,5)*100
    ###Acc and Loss
    epoch_loss = np.mean(epoch_loss)
    return epoch_loss, acc

############################################################################################################################################################################################################
############################################################################################################################################################################################################

def af_pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)

def collate_fn(batch):
    spectrograms,rgb_frames,labels = [], [], []
    # Gather in lists, and encode labels as indices
    for spec,frame,label in batch:
        spectrograms += [spec]
        rgb_frames += [frame]
        labels += [torch.tensor(label)]

    # Group the list of tensors into a batched tensor
    spectrograms = af_pad_sequence(spectrograms)
    rgb_frames = torch.stack(rgb_frames)
    labels = torch.stack(labels)
    return spectrograms,rgb_frames,labels

############################################################################################################################################################################################################
############################################################################################################################################################################################################

def train_test(args):

    train_dataset = AV_Dataset('train.csv',args.audio_dir,args.visual_dir)
    test_dataset = AV_Dataset('test.csv',args.audio_dir,args.visual_dir)
    trainloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=True, num_workers=16)
    testloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate_fn, shuffle=False, num_workers=16)
    print("\t Dataset Loaded")

    model = AVmodel(num_classes=args.num_classes, num_latent=args.num_latent, dim=args.adapter_dim)
    model.to(args.device)
    print("\t Model Loaded")
    print('\t Trainable params = ',sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    # Loss Function
    loss_fn = nn.CrossEntropyLoss()

    best_val_acc = []

    print("\t Started Training")
    for epoch in range(args.num_epochs):
        
        ### Training
        loss, acc = train_one_epoch(trainloader,model,optimizer,loss_fn,args.device)
        ### Validation
        val_loss, val_acc = val_one_epoch(testloader,model,loss_fn,args.device)

        print('\nEpoch....', epoch + 1)
        print("Training loss & accuracy......",round(loss,4), round(acc,3))
        print("Val loss & accuracy......", round(val_loss,4), round(val_acc,3))
        best_val_acc.append(val_acc)

    print("\n\t Completed Training \n")  
    print("\t Best Results........", np.max(np.asarray(best_val_acc)))


if __name__ == "__main__":
    opts = parse_options()
    train_test(args=opts)
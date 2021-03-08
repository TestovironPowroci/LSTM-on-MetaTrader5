import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os


# Parameters -----------------
epochs = 5000
firststart = True
lr = 0.001
bidirectional = True
dropout = 0.1
seq_lengths = 18
input_size = 4
hidden_size = 2
n_layers = 2
BATCH_SIZE = batch_size = 256
output_size = 3
spread = 1.1  # commission

count = 0
close = False


def check_correct(a, b):
    if a == 0:
        reward = 0
    if a == 2:

        if b == 2 or b == 3:
            reward = 1
        else:
            reward = -1
    if a == 1:
        if b == 1 or b == 3:
            reward = 1
        else:
            reward = -1
    return reward


def kalkulacja(out, targ):
    if out == 0:
        prof = 0
    if out == 2:
        prof = targ - spread
    if out == 1:
        prof = -targ - spread
    return prof


def load_checkpoint(model, optimizer, filename='state.pth'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch


class TrainDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('learning dax30.csv',
                        delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_datai = torch.from_numpy(xy[:, :-1])
        self.y_datai = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_datai[index], self.y_datai[index]

    def __len__(self):
        return self.len


TrainDataset = TrainDataset()
train_loader = DataLoader(dataset=TrainDataset,
                          batch_size=batch_size,
                          shuffle=True, pin_memory=True, drop_last=True)


class TestDataset(Dataset):

    def __init__(self):
        xy = np.loadtxt('test dax30.csv',
                        delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_datai = torch.from_numpy(xy[:, :-1])
        self.y_datai = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_datai[index], self.y_datai[index]

    def __len__(self):
        return self.len


TestDataset = TestDataset()
test_loader = DataLoader(dataset=TestDataset,
                         batch_size=1,
                         shuffle=False, pin_memory=True)


class ValidationDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('validation dax30 pips.csv',
                        delimiter=',', dtype=np.float32)

        self.len = xy.shape[0]
        self.x_datai = torch.from_numpy(xy[:, :-1])
        self.y_datai = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_datai[index], self.y_datai[index]

    def __len__(self):
        return self.len


ValidationDataset = ValidationDataset()
test_loaderTest = DataLoader(dataset=ValidationDataset,
                             batch_size=1,
                             shuffle=False, pin_memory=True)


class LSTMClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.rnn = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.batch_size = None
        self.hidden = None

    def forward(self, x):
        h0, c0 = self.init_hidden(x)
        out, (hn, cn) = self.rnn(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

    def init_hidden(self, x):
        h0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.n_layers, x.size(0), self.hidden_size)
        return [t for t in (h0, c0)]

        return y


classifier = LSTMClassifier(input_size, hidden_size, n_layers, output_size)

criterion = nn.CrossEntropyLoss()
opt = torch.optim.Adam(classifier.parameters(), lr=lr)

if firststart == False:
    load_checkpoint(classifier, opt)
    for g in opt.param_groups:
        g['lr'] = lr

for epoch in range(epochs):
    total_loss = 0
    classifier.train()
    for i, data in enumerate(train_loader):
        input2, target2 = data
        input2 = input2.view(batch_size, seq_lengths, input_size)
        target2 = target2.long().squeeze()
        classifier.zero_grad()
        output = classifier(input2)
        loss = criterion(output, target2)
        total_loss += loss.item()

        loss.backward()
        opt.step()

    average_loss = total_loss / (epoch + 1)

    print("epoch: ", epoch + 1, " LOSS: ", round(average_loss, 4))

    state = {'epoch': epoch + 1, 'state_dict': classifier.state_dict(),
             'optimizer': opt.state_dict()}

    torch.save(classifier.state_dict(), 'classifier.pth')
    torch.save(opt.state_dict(), 'optimizer.pth')
    torch.save(state, 'state.pth')
    if epoch % 20 == 0:
        classifier.eval()
        for epoch in range(1):

            total_effect = 0
            good_decisions = 0
            actions_taken = 0

            for i, data in enumerate(test_loader):
                input, target = data

                input = input.view(1, seq_lengths, input_size)
                target = target.long().squeeze()
                output = classifier(input)
                output = torch.argmax(output, dim=1).item()
                r = check_correct(output, target)
                total_effect += r

                if r == 1:
                    good_decisions += 1

                if r == -1 or r == 1:
                    actions_taken += 1

            print("TEST:  ", '|wynik ogólny: ', total_effect, "|good_decisions: ", good_decisions, "|all_decisions: ",
                  actions_taken)

    if epoch % 20 == 0:
        classifier.eval()
        for epoch in range(1):

            zysk = 0
            total_effect = 0
            good_decisions = 0
            actions_taken = 0
            count = 0

            for i, data in enumerate(test_loaderTest):
                input, target = data

                input = input.view(1, seq_lengths, input_size)
                target = target.long().squeeze()
                output = classifier(input)
                output = torch.argmax(output, dim=1).item()

                if close == False:
                    profit = kalkulacja(output, target)
                    zysk += profit

                    if profit > 0:
                        r = 1
                        good_decisions += 1
                        actions_taken += 1

                    if profit < 0:
                        r = -1
                        actions_taken += 1

                    if profit == 0:
                        r = 0

                    total_effect += r
                if profit != 0:
                    close = True
                    count += 1
                    if count >= 30:
                        close = False
                        count = 0

            print("WALIDACJA:   ", '|wynik ogólny: ', total_effect, "|good_decisions: ", good_decisions,
                  "all_decisions: ", actions_taken, "zysk", zysk)
            close = False

        if 120 < good_decisions:  # if profit is high enough, save
            if (good_decisions / actions_taken) > 0.6:
                torch.save(classifier.state_dict(), 'path/classifier120.pth')
                torch.save(opt.state_dict(), 'path/optimizer120.pth')
                torch.save(state, 'path/state120.pth')
                print("zapis formuły.")

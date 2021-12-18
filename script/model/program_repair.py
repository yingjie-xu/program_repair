import json
import time
import math
import os
import argparse
import torch
import torch.nn as nn
from model import LSTM
from data_clean import randomTrainingExample, inputTensor
from constant import all_letters, n_letters


def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = lstm.initHidden()
    cell = lstm.initCell()

    lstm.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden, cell = lstm.forward(input_line_tensor[i], hidden, cell)
        print(target_line_tensor[i])
        print(output.size())
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in lstm.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='./data/hw0.json')
    parser.add_argument('--output_dir', type=str, default='./res/hw0')
    args = parser.parse_args()
    file = args.file
    output_dir = args.output_dir
    file_name = file.split('/')[-1]
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)


    with open(file, 'r') as fd:
        data = json.load(fd)

    criterion = nn.NLLLoss()

    learning_rate = 0.0005

    lstm = LSTM(n_letters, 512, 512, n_letters)

    n_iters = 10
    print_every = 1
    plot_every = 1
    all_losses = []
    total_loss = 0  # Reset every plot_every iters

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample(data))
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' %
                  (timeSince(start), iter, iter / n_iters * 100, loss))
            torch.save(lstm, os.path.join(output_dir, f'model_at_{iter}.pt'))
            # torch.save(lstm, f'./{file_name}/model_at_{iter}.pt')

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0

import json
import time
import math
import os
import argparse
import torch
import torch.nn as nn
from model import LSTM
from data_clean import random_training_batch, inputTensor
from constant import all_letters, n_letters


def train(input_line_tensor, target_line_tensor):
    hidden = lstm.initHidden(batch_size)
    cell = lstm.initCell(batch_size)

    lstm.zero_grad()

    loss = 0

    for i in range(input_line_tensor.size(1)):
        input_ts = input_line_tensor[:, i, :, :]
        output, hidden, cell = lstm.forward(input_ts, hidden, cell)
        l = criterion(output, target_line_tensor[:, i])
        loss += l
        # [[onehot, 0, 0, 1], [1, 0, 0, 0]] -> [[0, 0, 0, 1], [1, 0, 0]] -> [4, 3]
    loss.backward()

    for p in lstm.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / batch_size


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
    parser.add_argument('--num_iterations', type=str, default=10000)
    parser.add_argument('--print_every', type=str, default=1000)
    parser.add_argument('--batch_size', type=str, default=10)
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

    batch_size = int(args.batch_size)

    lstm = LSTM(n_letters, 512, 512, n_letters)

    n_iters = int(args.num_iterations)
    print_every = int(args.print_every)
    # plot_every = 1000
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    interval_loss = 0

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = train(*random_training_batch(data, batch_size))
        total_loss += loss
        interval_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' %
                  (timeSince(start), iter, iter / n_iters * 100, interval_loss / print_every))
            interval_loss = 0
            torch.save(lstm, os.path.join(output_dir, f'model_at_{iter}.pt'))

import random
import torch
import json
from constant import all_letters, n_letters


def randomChoice(data):
    return data[random.randint(0, len(data) - 1)]


# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        if all_letters.find(letter) == -1:
            print(f'Can\'t find {letter}')
        tensor[li][0][all_letters.find(letter)+1] = 1
    return tensor


def targetTensor(line):
    letter_indexes = [all_letters.find(
        line[li])+1 for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def randomTrainingExample(data):
    line = randomChoice(data)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor


def random_training_batch(data, N):
    input_ts, target_ts = [], []
    for i in range(N):
        input_tensor, target_tensor = randomTrainingExample(data)
        input_ts.append(input_tensor)
        target_ts.append(target_tensor)
    max_len = max([i.size()[0] for i in input_ts])
    for i in range(len(input_ts)):
        target = torch.zeros(max_len, 1, n_letters)
        target[:input_ts[i].size()[0], :, :] = input_ts[i]
        input_ts[i] = target
        target = torch.zeros(max_len)
        target[:target_ts[i].size()[0]] = target_ts[i]
        target_ts[i] = target
    input_ts = torch.stack(input_ts)
    target_ts = torch.stack(target_ts)
    target_ts = target_ts.type(torch.LongTensor)
    return input_ts, target_ts


if __name__ == "__main__":
    with open('../../data/result/hw2.json', 'r') as fd:
        data = json.load(fd)

    input_ts, target_ts = random_training_batch(data, 20)
    print(input_ts.size())
    # print(input_ts[1])
    print(target_ts)

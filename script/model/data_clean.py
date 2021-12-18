import random
import torch
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
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor


def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1)  # EOS
    return torch.LongTensor(letter_indexes)


def randomTrainingExample(data):
    line = randomChoice(data)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return input_line_tensor, target_line_tensor

# def random_training_batch(N):
#     input_ts, target_ts = [], []
#     for i in range(N):
#         input_tensor, target_tensor = randomTrainingExample()
#         input_ts.append(input_tensor)
#         target_ts.append(target_tensor)
#     x = torch.stack(input_ts, dim=0)
#     return x

# ex = random_training_batch(3)
# print(ex.size())
# print(ex[0].size())
# print(ex[1].size())

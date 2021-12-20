import torch
import torch.nn as nn
from data_clean import inputTensor
from constant import all_letters, n_letters


def sample(codes):
    with torch.no_grad():  # no need to track history in sampling
        hidden = model.initHidden(1)
        cell = model.initCell(1)
        output = None

        for i in codes:
            it = inputTensor(i)
            output, hidden, cell = model(it[0].unsqueeze(0), hidden, cell)
            topv, topi = output.topk(1)
            if topi == n_letters - 1:
                return 'END'
            else:
                letter = all_letters[topi-1]
                output = letter
        return output


if __name__ == '__main__':
    model = torch.load('./res/hw2/model_at_1.pt')
    print(sample(
        "let partition (p : 'a -> bool) (l : 'a list) : ('a list * 'a list) = let f i (j"))
    print(sample("let partition (p : 'a -> bool) (l : 'a list) : ('a list * 'a list) = let helper e (t,f) = if p e then (e::t"))
    print(sample("let partition (p : 'a -> bool) (l : 'a list) : ('a list * 'a list) = let f i (j,k)= if p i then (i::j"))

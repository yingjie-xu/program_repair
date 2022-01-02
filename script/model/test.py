import torch
import torch.nn as nn
import json
from data_clean import inputTensor
from constant import all_letters, n_letters


class Test:
    def __init__(self):
        self.model = None

    def sample(self, codes):
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        with torch.no_grad():  # no need to track history in sampling
            hidden = self.model.initHidden(1).to(device)
            cell = self.model.initCell(1).to(device)
            output = None

            for i in codes:
                it = inputTensor(i).to(device)
                output, hidden, cell = self.model(
                    it[0].unsqueeze(0), hidden, cell)
                topv, topi = output.topk(1)
                if topi == n_letters - 1:
                    return 'END'
                else:
                    letter = all_letters[topi-1]
                    output = letter
            return output

    def test_hw0(self):
        log = []
        log.append("====testing hw0====")
        with open('./test_data/test_hw0.json') as fd:
            dic = json.load(fd)
        self.model = torch.load('./res/hw0_batch_10/model_at_39000.pt')
        correct, total = 0, 0
        for key, value in dic.items():
            prediction = self.sample(key)
            log.append(
                f'prediction: "{prediction}" <-> truth: "{value}" with sample "{key}"')
            if prediction == value:
                correct += 1
            total += 1
        log.append(f'hw0 accuracy: {correct/total}')
        with open('./test_data/result/hw0_result', 'w') as fd:
            fd.write('\n'.join(log))

    def test_hw2(self):
        log = []
        log.append("====testing hw2====")
        with open('./test_data/test_hw2.json') as fd:
            dic = json.load(fd)
        self.model = torch.load('./res/hw2_batch_10/model_at_23000.pt')
        correct, total = 0, 0
        for key, value in dic.items():
            prediction = self.sample(key)
            log.append(
                f'prediction: "{prediction}" <-> truth: "{value}" with sample "{key}"')
            if prediction == value:
                correct += 1
            total += 1
        log.append(f'hw0 accuracy: {correct/total}')
        with open('./test_data/result/hw2_result', 'w') as fd:
            fd.write('\n'.join(log))


if __name__ == '__main__':
    test = Test()
    test.test_hw0()
    test.test_hw2()

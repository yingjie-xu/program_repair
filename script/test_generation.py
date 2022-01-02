import json


def generate_test_cases(input_path, output_path):
    with open(input_path, 'r') as fd:
        data = json.load(fd)
    dic = {}
    s = 'let distance_tests = [ ( ((0, 0), (3, 4)), (5.) );'
    for s in data:
        if ',' in s:
            line = s.split(',')
            for i in range(1, len(line)):
                cur = ','.join(line[:i])
            dic[cur] = ','
    with open(output_path, 'w') as fd:
        json.dump(dic, fd)



if __name__ == "__main__":
    generate_test_cases('../data/result/hw0.json',
                        '../data/result/test_hw0.json')
    generate_test_cases('../data/result/hw2.json',
                        '../data/result/test_hw2.json')

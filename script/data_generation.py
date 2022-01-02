import os
from utils import fetch_file_history, find_file_path, data_cleaning, write_result


def generate_hw0():
    path = find_file_path('../data/sync/')
    hw0_path = list(filter(lambda x: 'hw0' in x, path))
    hw0_path = hw0_path[:100]
    hw0_data = []
    for p in hw0_path:
        hw0_data.extend(data_cleaning(p))
    write_result(hw0_data, '../data/result/hw0.json')


def generate_hw2():
    path = find_file_path('../data/sync/')
    hw2_path = list(filter(lambda x: 'hw2' in x, path))
    # hw2_path = hw2_path[:100]
    hw2_data = []
    for p in hw2_path:
        hw2_data.extend(data_cleaning(p))
    write_result(hw2_data, '../data/result/hw2.json')


def generate_hw3():
    path = find_file_path('../data/sync/')
    hw2_path = list(filter(lambda x: 'hw3' in x, path))
    # hw2_path = hw2_path[:100]
    hw2_data = []
    for p in hw2_path:
        hw2_data.extend(data_cleaning(p))
    write_result(hw2_data, '../data/result/hw3.json')


if __name__ == "__main__":
    generate_hw3()

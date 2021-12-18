from git import Repo
from collections import defaultdict
import os
import json


def fetch_file_history(repo_path='./', branch_name='master', output_path='./file_versions', file_types=['ml']):
    """Fetch the history of git commits from the repo_path,
        generate all the history files to the output_path

    Args:
        repo_path (str, optional): path to the repo. Defaults to './'.
        branch_name (str, optional): branch of the repo. Defaults to 'master'.
        output_path (str, optional): place to store the file history. Defaults to './file_versions'.
        file_types (list, optional): file types to generate the history. Defaults to ['ml'].
    """
    repo = Repo(repo_path)
    commits = list(repo.iter_commits(branch_name))
    commits.reverse()  # reverse the commit - start from beginning
    file_versions = defaultdict(lambda: 0)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for commit in commits:
        files = repo.git.show("--pretty=", "--name-only",
                              commit.hexsha).split('\n')
        files = list(filter(lambda file: file.split('.')
                     [-1] in file_types, files))
        for file in files:
            version = file_versions[file]
            file_versions[file] += 1
            file_content = commit.repo.git.show(f"{commit.hexsha}:{file}")
            file = file.split('.')
            file_name = f'{file[0]}_v{version}'
            print(f'Writing file: {file_name}')
            f = open(os.path.join(output_path, file_name), "w")
            f.write(file_content)
            f.close()


def data_cleaning(file_name):
    """remove comments and blank lines from ocaml files and separate them into functions

    Args:
        file_name (str): file to clean

    Returns:
        List[str]: list of functions returned from the file
    """
    with open(file_name) as file:
        lines = file.readlines()
    is_comment = False  # check if current line is a comment
    lines_without_comments = []
    for line in lines:
        if line == '\n':
            continue
        if '(*' in line:
            is_comment = True
        if not is_comment:
            lines_without_comments.append(line)
        if '*)' in line:
            is_comment = False
    functions = ['']
    for line in lines_without_comments:
        functions[-1] += f'{line[:-1].strip()} '
        if ';;' in line:
            functions.append('')
    functions.pop()  # remove the extra at the end
    return functions
    # print(len(functions))


def find_file_path(root_repo='.'):
    """find all the repos with full ocaml submission, and return path to those files

    Args:
        root_repo (str, optional): root repo path. Defaults to '.'.

    Returns:
        list: a list of all the file path
    """
    path = []
    all_root = []
    for root, dirs, files in os.walk(root_repo):
        if '.git' in dirs:
            dirs.remove('.git')
        if root == '.' or len(files) <= 6:
            continue
        files = filter(lambda x: '.ml' in x, files)
        path.extend([os.path.join(root, file) for file in files])
        all_root.append(root)
    return path


def write_result(data, file_name='clean.json'):
    """write the results to json

    Args:
        data (List): data to write
        file_name (str, optional): file name to write to. Defaults to 'clean.json'.
    """
    with open(file_name, 'w') as file:
        json.dump(data, file)


if __name__ == '__main__':
    path = find_file_path('../data/sync/')
    hw0_path = list(filter(lambda x: 'hw0' in x, path))
    hw0_path = hw0_path[:10]
    hw0_data = []
    for p in hw0_path:
        hw0_data.extend(data_cleaning(p))
    # for d in data:
    #     print(d)
    write_result(hw0_data, '../data/result/clean.json')

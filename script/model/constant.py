import string


all_letters = string.ascii_letters + \
    "1234567890 .,:;'_=[]()+-*\\\n<>/~!@#$%^&|}{\""
# 0 means nothing, EOS marker is the last char
n_letters = len(all_letters) + 2


if __name__ == "__main__":
    print(n_letters)
    print(all_letters)

import string


all_letters = string.ascii_letters + \
    "1234567890 .,:;'_=[]()+-*\\\n<>/~!@#$%^&|}{"
n_letters = len(all_letters) + 1  # Plus EOS marker

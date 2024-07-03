import os

ROOT = os.path.dirname(os.path.abspath(__file__))

def labels():
    with open(os.path.join(ROOT, '../data/labeled/classname.txt'), 'r') as file:
        lables = {str(i): row.strip() for i, row in enumerate(file, start=0)}
    return lables

import numpy as np
import pandas as p
from zipfile import ZipFile
import os

# print(os.getcwd())


with ZipFile('test_files/charity.zip') as zip:
    print(zip)




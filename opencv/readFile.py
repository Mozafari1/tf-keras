import numpy as np

# from itertools import zip_longest
import numpy.lib.arraysetops as aso

f = open("diff/count1.txt", "r")
f2 = open("diff/count1.txt", "r")
f1 = f.read()
f3 = f.read()


# def find_first_diff(f1, f3):
#     for i, (x, y) in enumerate(zip_longest(f1, f3, fillvalue=object())):
#         if x != y:
#             print(i)


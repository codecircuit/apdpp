import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

f = open("grid.txt", 'r')

content = f.read()
f.close()

x = [[]]

i = 0
j = 0

print content.split(' ')

for l in map(lambda s: s.split('\n') ,content.split(' ')):
    print l

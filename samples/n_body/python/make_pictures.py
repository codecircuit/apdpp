#!/usr/bin/python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import os
import csv

pthData = "output.csv"
pthPic = "pic/picture"
figsize = (7,7)
xlim = (-1,1)
ylim = (-1,1)
zlim = (-1,1)

f = open(pthData, 'r')
reader = csv.reader(f, delimiter=' ', quotechar='#')
i = 0

while 1:
    try:
        pos_x = reader.next()
        pos_y = reader.next()
        pos_z = reader.next()
        reader.next() # skip velocities
        reader.next()
        reader.next()
        masses = reader.next()
        try:       
            pos_x = map(float, pos_x)
        except ValueError:
            print "Could not convert", pos_x
        try:       
            pos_z = map(float, pos_z)
        except ValueError:
            print "Could not convert", pos_z
        try:       
            pos_y = map(float, pos_y)
        except ValueError:
            print "Could not convert", pos_y
        try:       
            masses = map(float, masses)
        except ValueError:
            print "Could not convert", masses
        masses = map(lambda x: x * 20, masses)

        pth = pthPic + "{:0>4d}".format(i) + ".png" 
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(pos_x, pos_y, pos_z, s=masses)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        print "Saving picture", pth
        fig.savefig(pth, dpi=200)
        plt.close()
        i += 1
    except StopIteration:
        break

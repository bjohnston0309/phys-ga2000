#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

x_data =  np.arange(-10,10,1e-5)
y_data = (1/np.sqrt(2*np.pi*(3**2)))*np.exp(-(x_data**2/(2*(3**2))))


plt.plot(x_data, y_data)
plt.xlabel('$x$ (m)', fontsize=16)
plt.ylabel('$y$ (Hz)', fontsize=16)
plt.title('Gaussian Plot', fontsize=20)
plt.savefig('Gaussian.png')






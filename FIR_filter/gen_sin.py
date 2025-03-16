import numpy as np
from scipy.io import wavfile
from matplotlib import pyplot
fs = 10000

samples = np.arange(0,2,1/fs)
samples = np.arange(0,2,1/fs)


signal = 0.3*np.sin(2 * np.pi *200  * samples)+ 0.5*np.sin(2 * np.pi * 1500  * samples) 

signal *= 65536
signal += 65536
plt = pyplot.plot(signal)
pyplot.savefig('fig.png')
signal = np.int32(signal)

f = open('FIR_filter/sin.txt','w+')
f2 = open('FIR_filter/sin2.txt','w+')
for i in signal:
    ib = 0
    if i >= 0:
        ib = format(i,'b').zfill(32)
       
    if i < 0:
        ib = format((pow(2,32)-abs(i)),'b').zfill(32)
    ib2 = format(i,'x')
    f.write((ib)+'\n')
    f2.write(str(ib2)+'\n')

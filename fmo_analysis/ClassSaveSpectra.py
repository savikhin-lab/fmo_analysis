import os
import os.path
import numpy as np
import time
import copy
import string

class Spectrum:
    def __init__(self,spectrum = [[],[]],comment = '',settime = 0):
        self.spectrum = spectrum
        self.comment = comment
        if settime == 0:   self.time = time.time()
        else:              self.time = settime
            

class MultiSpectraFile:
    def __init__(self,filename='', openmode = 'w'):
        self.filename = filename
        self.openmode = openmode
        self.spectra = []
        return
    
    def append(self,spectrum):
        self.spectra.append(copy.deepcopy(spectrum))
        return
    
    def save(self):
        f = open(self.filename,self.openmode)
        for s in self.spectra:
            f.write('##### SPECTRUM #####\n')
            lins = s.comment.splitlines()
            f.write('#COMMENT\n%d\n' % len(lins))
            for l in lins:
                f.write(l + '\n')
            f.write('#TIME\n%17.15g\n' % s.time)
            f.write('#TIMESTRING\n' + time.strftime('%Y.%m.%d,%H:%M:%S',time.localtime(s.time))+'\n')
            le = min(len(s.spectrum[0]),len(s.spectrum[1]))
            f.write('#DATA\n%d\n' % le)
            for i in range(0,le):
                f.write('%g,%g\n' % (s.spectrum[0][i],s.spectrum[1][i]))
            a = 1
        f.close()
        return
        
"""
x = [1,2,3,4,5]
y = [10,20,30,40,50]
comment = 'first line\nsecond line'

s = Spectrum([x,y],comment)

fil = MultiSpectraFile(filename='test.dat',openmode='w')
fil.append(s)

s.comment = 'second spectrum\nhaha'

fil.append(s)
fil.save()
"""
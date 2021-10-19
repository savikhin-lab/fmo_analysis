#import numpy as np
#import math
#from openpyxl import Workbook
#from openpyxl import load_workbook
#from openpyxl.chart import (ScatterChart, Reference, Series)
from ClassLoadDataFiles import Snapshots
from ClassSaveSpectra import Spectrum
from ClassSaveSpectra import MultiSpectraFile
import matplotlib.pyplot as plt
import time
import numpy as np
import os
import os.path
import copy
import math
import random as random
from ClassExciton_dual import Exciton
from ClassExciton_dual import Pigment
from Exciton_mutant_subs import Params
from Exciton_mutant_subs import Routine
from Exciton_mutant_subs import save_matrix
from Exciton_mutant_subs import load_matrix
from Exciton_mutant_subs import plot
from Exciton_mutant_subs import load_xy

# load experimental curves
xabs3,abs3 = load_xy('Experiment-mutant3-dOD.csv')
xcd3,cd3 = load_xy('Experiment-mutant3-dCD.csv')
xabs7,abs7 = load_xy('Experiment-mutant7-dOD.csv')
xcd7,cd7 = load_xy('Experiment-mutant7-dCD.csv')

pars = Params()
pars.bandwidth = 70
pars.scale=False
#pars.shift_diag += 30

directory_wt = 'Hamiltonian_2020_05_23_YB/rotated'
directory_3 = 'Hamiltonian-mutant3_2021_06_09'
directory_7 = 'Hamiltonian-mutant7_2021_06_09'
# assume average hamiltonian and coordinates in file /rotated/average

# Do wild type first
wt = Routine()
wt.Doall(directory_wt,pars)
wt.get_average_ham(pars)
pars.delete_pig8 = True
wt_7 = Routine()
wt_7.Doall(directory_wt,pars)
pars.delete_pig8 = False

mutant3 = Routine()
mutant3.Doall(directory_3,pars)
mutant3.get_average_ham(pars)
pars.delete_pig8 = True
mutant3_7 = Routine()
mutant3_7.Doall(directory_3,pars)
pars.delete_pig8 = False


mutant7 = Routine()
mutant7.Doall(directory_7,pars)
mutant7.get_average_ham(pars)
pars.delete_pig8 = True
mutant7_7 = Routine()
mutant7_7.Doall(directory_7,pars)
pars.delete_pig8 = False

frac = .55
abs_wt = wt.ab*frac + wt_7.ab*(1.-frac)
cd_wt = wt.cd*frac + wt_7.cd*(1.-frac)
abs_m3 = mutant3.ab*frac + mutant3_7.ab*(1.-frac)
cd_m3 = mutant3.cd*frac + mutant3_7.cd*(1.-frac)
abs_m7 = mutant7.ab*frac + mutant7_7.ab*(1.-frac)
cd_m7 = mutant7.cd*frac + mutant7_7.cd*(1.-frac)


save_matrix(directory_3+'/dif_ham_average.txt',mutant3.exa.ham - wt.exa.ham)
save_matrix(directory_7+'/dif_ham_average.txt',mutant7.exa.ham - wt.exa.ham)
save_matrix(directory_wt+'/ham_average_shift'+str(pars.shift_diag)+'.txt',wt.exa.ham)
shifted = np.array(wt.exa.ham)
for i in range(0,8):
    shifted[i][i] = 1e7/shifted[i][i]
save_matrix(directory_wt+'/ham_average_shift'+str(pars.shift_diag)+'_nm.txt',shifted)

x = wt.x

plot('Yongbin: m3 (blue), m7 (orange), wt(green)',
     [[[x,abs_m3],[x,abs_m7],[x,abs_wt]],
      [[x,cd_m3],[x,cd_m7],[x,cd_wt]]])

plot('Yongbin: m3-wt(blue), experiment (orange))',
     [[[x,abs_m3-abs_wt],[xabs3,abs3*8.]],
      [[x,cd_m3 - cd_wt],[xcd3,cd3*4]]])

plot('Yongbin: m7-wt(blue), experiment (orange))',
     [[[x,abs_m7-abs_wt],[xabs7,abs7*20.]],
      [[x,cd_m7 - cd_wt],[xcd7,cd7*5]]])
      
diag_only = True #shift only diagonals

mat3 = load_matrix(directory_3+'/dif_ham_average.txt')
mat7 = load_matrix(directory_7+'/dif_ham_average.txt')
#mat7[7][7] = 80

bw = 100

for name in ['HamiltonBrixner.xlsx','HamiltonKell.xlsx']:
    ex = Exciton(filename = name)
    ex.bandwidth = bw
    ex.getsticks()
    ex.getspectra()
    
    ex3 = Exciton(filename = name)
    
    for i in range(0,ex.size):
        if diag_only:
            ex3.ham[i][i] += mat3[i][i]
        else:
            for j in range(0,ex.size):
                ex3.ham[i][j] += mat3[i,j]
    ex3.getsticks()
    ex3.bandwidth = bw
    ex3.getspectra()
    
    ex7 = Exciton(filename = name)
    
    for i in range(0,ex.size):
        if diag_only:
            ex7.ham[i][i] += mat7[i][i]
        else:
            for j in range(0,ex.size):
                ex7.ham[i][j] += mat7[i,j]
    ex7.getsticks()
    ex7.bandwidth = bw
    ex7.getspectra()
    
    x = 1e7/ex.x
    plot(name+': Mt3 blue, M7 orange, wt green',
         [[[x,ex3.abs],[x,ex7.abs],[x,ex.abs]],
          [[x,ex3.CD],[x,ex7.CD],[x,ex.CD]]])
    
    plot(name+': M3-wt: calc blue, experiment orange',
         [[[x,ex3.abs-ex.abs],[xabs3,abs3*4.]],
          [[x,ex3.CD - ex.CD],[xcd3,cd3/2]]])
    
    plot(name+': m7-wt: calc blue, experiment orange',
         [[[x,ex7.abs-ex.abs],[xabs7,abs7*5.]],
          [[x,ex7.CD - ex.CD],[xcd7,cd7/1.5]]])



"""
##########################################
Calculate average population of each pigment at T=temp for Boltzman distribution
pigpop[i] = population
############################################
"""

"""
i=0
pigpop = np.zeros((ex.size)) # pigment population for excitons
pigpol = np.zeros((ex.size)) # population for each pigment assuming NO delocalization
temp = 297  # temperature
kt = temp/300*0.02585   # kT in eV
for ex in exciton:
    pops = np.zeros((ex.size))
    pols = np.zeros((ex.size))
    for k in range(0,ex.size):
        de = (ex.eval[k] - ex.eval[0])*0.000124 # convert wavenumbrs to eV
        pop = math.exp(-de/kt)   # population of this state
        psum=0
        for m in range(0,ex.size):
            pops[m] = pops[m] + pop*ex.evec[m,k]**2
        if delete_pig8: pops[7]=0
        de = (ex.ham[k,k] - ex.ham[3,3])*0.000124
        pol = math.exp(-de/kt)
        pols[k] = math.exp(-de/kt)
    pops = pops/np.sum(pops)
    pigpop = pigpop + pops
    if delete_pig8: pols[7] = 0.
    pols = pols/np.sum(pols)
    pigpol = pigpol + pols
print(pigpop)
pigpop=pigpop/np.sum(pigpop)
pigpol=pigpol/np.sum(pigpol)
print("Population at "+str(temp)+'K:')
print("coupling on:")
print(pigpop)
print("coupling off:")
print(pigpol)
"""   


"""
###########################################
#averaging hamiltonians and calculating spectra for averaged hamiltonian
############################################
"""
"""
count=0
for ex in exciton:
    if count==0:
        exa = copy.deepcopy(ex)
    else:
        exa.ham = exa.ham+ex.ham
        for i in range(0,ex.size):
            exa.pig[i].coord = exa.pig[i].coord + ex.pig[i].coord
            exa.pig[i].mu =  exa.pig[i].mu + ex.pig[i].mu
            exa.pig[i].mag = exa.pig[i].mag + ex.pig[i].mag
    count = count + 1
exa.ham = exa.ham/count
for i in range(0,ex.size):
    exa.pig[i].coord = exa.pig[i].coord/count
    exa.pig[i].mu =  exa.pig[i].mu/count
    exa.pig[i].mag = exa.pig[i].mag/count
exa.bandwidth = math.sqrt(bandwidth**2 + 90**2)
exa7=copy.deepcopy(exa)
exa.getsticks()
exa.getspectra()
exa7.getsticks()
exa7.getspectra()

absa=exa.abs*0.55 + exa7.abs*0.45
cda=exa.CD*0.55 + exa7.CD*0.45
absa=exa.abs
cda=exa.CD

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
ax1.plot(x, absa)
ax2.plot(x,cda)

ax1.set(xlabel='', ylabel='absorbance',title='Exciton of average')
ax2.set(xlabel='wavenumers', ylabel='CD')
ax1.grid()
ax2.grid()
fig.savefig('AverageHam-'+directory+'-'+str(int(exa.bandwidth))+'.jpg')
"""

"""
#####################
Print average Hamiltonian and width distribution of each parameter in cm-1, diagonal and coupling
######################
"""
"""
d_ham = np.zeros((8,8)) # rms from avegage
count=0
for ex in exciton:
    d_ham = d_ham + (ex.ham - exa.ham)**2
    count = count +1
d_ham = d_ham/count
d_ham = np.sqrt(d_ham)
na=directory+'/Hamiltonian-rms.csv'
filemat = open(na,'w')
for i in range(0,8):
    for j in range(0,8):
        if i==j:
            filemat.write("%.0f±%.0f" % (exa.ham[i][j],d_ham[i][j]))
        if i<j:
            filemat.write("%.2f" % exa.ham[i][j])
        if i>j:
            filemat.write("±%.2f" % d_ham[i][j])
        if j<7:
            filemat.write(",")
    filemat.write("\n")
filemat.close()
"""

"""
Calculate pigment population for average Hamiltonian
"""
"""
ex=exa
pops = np.zeros((ex.size))
pols = np.zeros((ex.size))
for k in range(0,ex.size):
    de = (ex.eval[k] - ex.eval[0])*0.000124 # convert wavenumbrs to eV
    pop = math.exp(-de/kt)   # population of this state
    psum=0
    for m in range(0,ex.size):
        pops[m] = pops[m] + pop*ex.evec[m,k]**2
    if delete_pig8: pops[7]=0
    de = (ex.ham[k,k] - ex.ham[3,3])*0.000124
    pol = math.exp(-de/kt)
    pols[k] = math.exp(-de/kt)
pops = pops/np.sum(pops)
if delete_pig8: pols[7] = 0.
pols = pols/np.sum(pols)
print("Populations for average Hamiltonian:")
print("couplings on:")
print(pops)
print("couplings off:")
print(pols)
"""

"""
###################################################
#average pigment energy positions
##############################################################
"""
"""
sigma2 = 200.
posav=np.zeros((8))
file_pig = open(nampig,'w')
file_pig.write("%g\n" % (8))
for i in range(0,8):
    count=0
    y = ex.x * 0
    posav[i]=0.
    for ex in exciton:
        y += np.exp(-(ex.x-ex.ham[i,i])**2/sigma2)
        posav[i]=posav[i]+ex.ham[i,i]
        count = count+1
    y = y/count
    posav[i]=posav[i]/count
    file_pig.write('Pigment '+str(i+1)+' position (sg2='+str(sigma2)+')\n%g\n0\n' % len(y))
    for j in range(0,len(x)):
        file_pig.write("%g,%g\n" % (x[j],y[j]))
   
   # plt.figure(figsize(100,50))
    fig, (ax1) = plt.subplots(1,1,) #, sharex=True)
    fig.set_figheight(1.5)
    fig.set_figwidth(8)
    ax1.plot(x, y)
    ax1.set(xlabel='', ylabel='absorbance',title='pigment '+str(i+1)+' aver position='+str(1e7/posav[i]))
    ax1.grid()

file_pig.close()
"""

"""
##############################################################
#Calculate correlations in pigment energy fluctuations
##############################################################
"""
"""
rms=np.zeros((8))   # rms**2 of each position
for i in range(0,8):
    for ex in exciton:
        rms[i]=rms[i]+(ex.ham[i,i]-posav[i])**2
rms /= count
distav=np.zeros((8,8))
#rms=np.sqrt(rms)
for i in range(0,7):
    for k in range(i+1,8):
        for ex in exciton:
            distav[i,k] = distav[i,k] + ex.ham[i,i]-ex.ham[k,k]
distav /= count
distrms=np.zeros((8,8))
for i in range(0,7):
    for k in range(i+1,8):
        for ex in exciton:
            distrms[i,k] = distrms[i,k]+(ex.ham[i,i]-ex.ham[k,k]-distav[i,k])**2
distrms=distrms/count

for i in range(0,7):
    for k in range(i+1,8):
        rms1=math.sqrt(distrms[i,k])
        rms2=math.sqrt(rms[i]+rms[k])
        print('%g-%g(%g): rms1=%g, rms2=%g' %(i+1,k+1,distav[i,k],rms1,rms2))
"""

"""
##############################################################
#Calculate average spectrum with specific width of each band
##############################################################
"""
"""
exa.x = np.arange(exa.xfrom,exa.xto,exa.xstep)
abscon = exa.x * 0.
cdcon = exa.x * 0.

for i in range(0,N):
    #sigma2 = exa.bandwidth**2/(4.*math.log(2.))
    sigma2 = 0.
    for j in range(0,N):
        sigma2 = sigma2 + ((exa.evec[j,i])**2) * rms[j]
    abscon += exa.stickA[i] * np.exp(-(exa.x-exa.eval[i])**2/(sigma2))
    cdcon += exa.stickCD[i] * np.exp(-(exa.x-exa.eval[i])**2/sigma2)

fig, (ax1,ax2) = plt.subplots(2,1,) #, sharex=True)
ax1.plot(x, abscon)
ax1.plot(x,exa.abs)
ax1.plot(x,ab/count*2)
ax2.plot(x,cdcon)
ax2.plot(x,exa.CD)
ax2.plot(x,cd/count*2)
ax1.set(xlabel='', ylabel='absorbance',title='average ham, variable rms')
ax1.grid()
ax2.set(xlabel='wavelength', ylabel='CD',title='')
ax2.grid()
"""

"""
##############################################################
shifting pigment energy position and calculate expected difference spectrum (mutants)
##############################################################
"""
"""
shiftpig=8-1    # numbering from 0!
shiftval=-60
count=0
for ex in exciton:
    ex.restoreham()
    ex.ham[shiftpig,shiftpig] = ex.ham[shiftpig,shiftpig] + shiftval
    ex.getsticks()
    ex.getspectra()
    if count == 0:
        abs1 = np.copy(ex.abs)
        cds1 = np.copy(ex.CD)
    else:
        abs1 = np.add(abs1,ex.abs)
        cds1 = np.add(cds1,ex.CD)
    ex.ham[shiftpig,shiftpig] = ex.ham[shiftpig,shiftpig] - 2*shiftval
    ex.getsticks()
    ex.getspectra()
    if count == 0:
        abs2 = np.copy(ex.abs)
        cds2 = np.copy(ex.CD)
    else:
        abs2 = np.add(abs2,ex.abs)
        cds2 = np.add(cds2,ex.CD) 
    count = count + 1

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
fig.set_figheight(2.5)
fig.set_figwidth(8)
ax1.plot(x, ab)
ax1.plot(x,abs1)
ax2.plot(x,cd)
ax2.plot(x,cds1)

ax1.set(xlabel='', ylabel='absorbance',title='Exciton spectrum shift '+str(shiftpig+1)+' by '+str(shiftval))
ax2.set(xlabel='wavelength, nm', ylabel='CD')
ax1.grid()
ax2.grid()
plt.show()

abs1 = np.subtract(abs1,ab)
cds1 = np.subtract(cds1,cd)
abs2 = np.subtract(abs2,ab)
cds2 = np.subtract(cds2,cd)

fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
fig.set_figheight(2.5)
fig.set_figwidth(8)
ax1.plot(x, abs1)
#ax1.plot(x, abs2)
ax2.plot(x, cds1)
#ax2.plot(x, cds2)

ax1.set(xlabel='', ylabel='absorbance difference, ',title='Shift '+str(shiftpig+1)+' by '+str(shiftval)+' cm^-1')
ax2.set(xlabel='wavelength, nm', ylabel='CD')
ax1.grid()
ax2.grid()
"""

"""
##############################################################
mix diagonals and coupolings randomly
##############################################################
"""
"""
for m in range(1,50000):
    e = random.randint(0,len(exciton)-1)
    f = random.randint(0,len(exciton)-1)
    i = random.randint(0,7)
    j = random.randint(0,7)
    a1 = exciton[e].ham[i,j]
    a2 = exciton[e].ham[j,i]
    exciton[e].ham[i,j] = exciton[f].ham[i,j]
    exciton[e].ham[j,i] = exciton[f].ham[j,i]
    exciton[f].ham[i,j] = a1
    exciton[f].ham[j,i] = a2

i=0
for ex in exciton:
    ex.getsticks()
    ex.getspectra()
    if i == 0:
        abm = np.copy(ex.abs)
        cdm = np.copy(ex.CD)
    else:
        abm = np.add(abm,ex.abs)
        cdm = np.add(cdm,ex.CD)    
    i = i+1
fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
#fig.set_figheight(2.5)
#fig.set_figwidth(8)
ax1.plot(x, ab)
ax1.plot(x, abm)
ax2.plot(x, cd)
ax2.plot(x, cdm)

ax1.set(xlabel='', ylabel='absorbance',title='mixing')
ax2.set(xlabel='wavenumers', ylabel='CD')
ax1.grid()
ax2.grid()
"""

"""
Take a bunch of snapshots into a movie into snap###.jpg
randomly remove pigment #8 with 55% probablility
"""
"""
x = 1.e7/exciton[0].x
aa=x*0.
ca=x*0.
for i in range(0,len(exciton)):
    ex = exciton[i]
    pig8=8
    if random.random() > 0.55:
        ex.delete_pigment(8)
        ex.getsticks()
        ex.getspectra()
        pig8=7
    a = exciton[i].abs
    c = exciton[i].CD
    fig, (ax1a,ax2a,ax1,ax2) = plt.subplots(4,1, sharex=True)
    fig.set_figheight(8)
    fig.set_figwidth(6)
    #fig.set_figheight(2.5)
    #fig.set_figwidth(8)
    ax1.plot(x, a)
    ax1.set(ylim=(0,4))
    ax2.plot(x, c)
    ax2.set(ylim=(-10,10))
    ax1.set(xlabel='', ylabel='absorbance',title='Snapshot '+str(i)+': '+str(pig8)+' pigments')
    ax2.set(xlabel='Wavelength, nm', ylabel='CD')
    ax1.grid()
    ax2.grid()
    aa = aa*float(i)
    aa = aa + a
    aa = aa/float(i+1)
    ca = ca*float(i)
    ca = ca + c
    ca = ca/float(i+1)
    #aa = aa/float(i)
    ax1a.plot(x,aa)
    ax1a.set(ylim=(0,2.5))
    ax1a.set(xlabel='', ylabel='absorbance',title='Average spectra')
    ax1a.grid()
    ax2a.plot(x,ca)
    ax2a.set(ylim=(-7,7))
    ax2a.set(xlabel='', ylabel='CD',title='')
    ax2a.grid()
    fig.savefig('Snap'+str(i)+'.jpg')
#plt.close('all')
"""
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
#import csv

def load_xy(filename):
    x = []
    y = []
    with open(filename,'r') as file:
        for line in file:
            s = line.split(",")
            x.append(float(s[0]))
            y.append(float(s[1]))
    return np.array(x),np.array(y)
            

def save_matrix(filename,mat):
    f = open(filename,'w')
    for i in range(0,len(mat[0])):
        for j in range(0,len(mat[0])):
            f.write("%10.5g " % mat[i][j])
        f.write('\n')

def load_matrix(filename):
    f = open(filename,'r')
    mat = np.zeros((8,8))
    i = 0
    for line in f:
        s = line.split()
        for j in range(0,8):
            mat[i][j] = float(s[j])
        i += 1
    return mat

def plot(title,data):
    frames = len(data)
    fig, ax = plt.subplots(frames,1, sharex=True)
    for i in range(0,frames):
        graphs = len(data[i])
        for j in range(0,graphs):
            ax[i].plot(data[i][j][0],data[i][j][1])
        ax[i].grid()
        ax[i].set(xlabel='', ylabel='',title=title)
                

class Params:
    def __init__(self):
        self.xfrom = 11790.
        self.xto = 13300.
        self.xstep = 1.
        self.bandwidth = 200
        self.shift_diag = -2420
        self.pignums = 8
        self.delete_pig8 = False
        self.dip_cor = 0.104
        self.delete_pig = 0
        self.use_shift_T = False
        self.scale = False
        self.ignore_offdiagonal_shifts = False # take off-diagonals in Yongbin 100 Hams as average, change diagonal only
    
class Routine:
    def __init__(self):
        self.pars = Params()
        
    def Doall(self,directory,pars):
        files = directory + '/files.txt'
        with open(files) as f:
            filenames = []
            for line in f: filenames.append(line.split()[0])
        Ne = len(filenames)
        print(filenames[0])
        # load and calculate exciton spectra for all N data files
        #ex = np.ndarray(Ne,Exciton)
        #ex = Exciton('HamiltonKell.xlsx','Temp.xlsx')
        #ex.bandwidth = 40
        #exnum = 0
        exciton = []
        i = 0
        lim1=pars.delete_pig
        lim2=pars.delete_pig+1
        TDM2_average = 0.# will get average TDM^2 for normalizing calculated spectra to TDM^2 = 1
        TDM2_count = 0
        
        if pars.delete_pig<0:
            lim1=0
            lim2=pars.pignums+1
            aball = []
            cdall = []
        for pars.delete_pigment in range(lim1,lim2):
            exciton = []
            TDM2_average = 0.# will get average TDM^2 for normalizing calculated spectra to TDM^2 = 1
            TDM2_count = 0
            i=0
            print('delpig='+str(pars.delete_pigment)+'\n')
            for nam in filenames:
                #print(nam)
                namh = directory + '/' + nam
                #check files
                if os.path.isfile(namh) == False:
                    print("Hamiltonian file does not exist, skip:")
                    print(namh)
                    continue
                if os.access(namh, os.R_OK) == False:
                    print("Hamiltonian file is not accessible, skip:")
                    print(namh)
                    continue     
                #load hamiltonian
                with open(namh) as f:
                    ham = []
                    for line in f: # read rest of lines
                        ham.append([float(x) for x in line.split()])
                # shift diagonal elements
                for m in range(0,len(ham)):
                    if len(ham) != len(ham[m])-6:
                        print('!!! HAMILTONIAN MUST BE SQUARE + mu(3) + pos(3)!!!')
                        print('File: ',namh)
                        print('# of lines = ',len(ham))
                        print('line ',m,' length = ',len(ham[m]))
                        exit()
                    ham[m][m]=ham[m][m]+pars.shift_diag    
                # load pigments
            
                pig = []
                N=len(ham)
                for m in range(0,N):
                    p = Pigment()
                    p.mu = np.copy([ham[m][N],ham[m][N+1],ham[m][N+2]])
                    p.coord =  np.copy([ham[m][N+3],ham[m][N+4],ham[m][N+5]])
                    TDM2_average = TDM2_average + np.dot(p.mu,p.mu)
                    TDM2_count = TDM2_count + 1
                    p.mag = 0.
                    pig.append(p)
                ha = np.zeros((N,N))
                for m in range(0,N):
                    for n in range(0,N):
                       ha[m,n] = ham[m][n]
                # construct an exciton and get spectra
            
                ex=Exciton()
                ex.setsystem(ha,pig) 
                if pars.delete_pig8 == True:
                    ex.delete_pigment(8)
                ex.delete_pigment(pars.delete_pigment)
                if pars.use_shift_T >0 :
                    if delete_pigment>0:
                        if use_shift_T == 2:
                            nams = directory + '/' + nam.replace('.csv','-shift.csv') # triplet shift file name
                        else:
                            nams = directory + '/confaverage_shifts.csv'
                        if os.path.isfile(nams) == False:
                            print('Shift file does not exist, skip:' + nams)
                            continue
                        if os.access(nams, os.R_OK) == False:
                            print('Shift file is not accessible, skip:' + nams)
                            continue
                        with open(nams) as f:
                            shift_T = []
                            for line in f: # read rest of lines
                                shift_T.append([float(x) for x in line.split()])
                        for ik in range(0,N):
                            ex.ham[ik][ik]=ex.ham[ik][ik] + shift_T[delete_pigment-1][ik]
                ex.setlimits(pars.xfrom,pars.xto,pars.xstep)
                ex.bandwidth = pars.bandwidth
                ex.getsticks()
                ex.getspectra()
                exciton.append(ex)      # for future, don't need to keep them all
                # average spectra
                if i == 0:
                    ab = np.copy(ex.abs)
                    cd = np.copy(ex.CD)
                else:
                    ab = np.add(ab,ex.abs)
                    cd = np.add(cd,ex.CD)
                i = i +1
            ab /= i
            cd /= i
            print('Loaded ',len(exciton),' hamiltonians out of ',Ne)
            TDM2_average = TDM2_average/TDM2_count
            print(f'Average TDM^2 ={TDM2_average}')
            if pars.scale:
                print(f' --- (all spectra divided by TDM2_average')
                ab /= TDM2_average
                cd /= TDM2_average
            p8='-8pig'
            if pars.delete_pig8 == True:
                p8 = '-7pig'
            namabs = directory+'/abs-'+str(pars.bandwidth)+'-'+str(pars.delete_pigment)+p8+'.xy'
            namcd = directory+'/cd-'+str(pars.bandwidth)+'-'+str(pars.delete_pigment)+p8+'.xy'
            nampig = directory+'/pig_pos'+'.dat'
            if pars.delete_pig<0:
                aball.append(ab)
                cdall.append(cd)
            x = 1e7/exciton[0].x
            file_abs = open(namabs,'w')
            file_cd = open(namcd,'w')
            for j in range(0,len(x)):
                file_abs.write("%g,%g\r\n" % (x[j],ab[j]))
                file_cd.write("%g,%g\r\n" % (x[j],cd[j]))
            file_abs.close()
            file_cd.close()
            fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
            ax1.plot(x, ab)
            ax2.plot(x, cd)
            
            ax1.set(xlabel='', ylabel='absorbance',title='Exciton spectrum-'+str(pars.delete_pigment)+directory)
            ax2.set(xlabel='wavelength, nm', ylabel='CD')
            ax1.grid()
            ax2.grid()
            #fig.savefig('Average-'+directory+'-'+str(pars.bandwidth)+'.jpg')
            #fig.savefig("test.png")
            plt.show()
        if pars.delete_pig<0:
            st=''
            if pars.use_shift_T>0:
                st = '-shift' + str(use_shift_T)
            namall = directory+'/allspectra-bw'+str(pars.bandwidth)+p8+st+'.dat'
            filall=open(namall,'w')
            filall.write("%g\n" % (pars.pignums*2+2))
            for i in range(0,pignums+1):
                text=str(i)+p8+st+'[FWHM='+str(exciton[0].bandwidth)+']-'+directory+'\n'+str(len(x))+'\n0\n'
                filall.write('ABS'+text)
                for k in range(0,len(x)):
                    j=len(x)-k-1
                    filall.write("%g,%g\n" % (x[j],aball[i][j]))
            for i in range(0,pars.pignums+1):
                text=str(i)+p8+st+'[FWHM='+str(exciton[0].bandwidth)+']-'+directory+'\n'+str(len(x))+'\n0\n'
                filall.write('CD'+text)
                for k in range(0,len(x)):
                    j=len(x)-k-1
                    filall.write("%g,%g\n" % (x[j],cdall[i][j]))
            filall.close()
            # Now record the #i-#0 difference spectra
            namall = directory+'/allspectra-bw'+str(bandwidth)+p8+st+'-diff.dat'
            filall=open(namall,'w')
            filall.write("%g\n" % (pars.pignums*2))
            for i in range(1,pars.pignums+1):
                text=str(i)+p8+st+'[FWHM='+str(exciton[0].bandwidth)+']-'+directory+'\n'+str(len(x))+'diff\n0\n'
                filall.write('ABS-diff'+text)
                for k in range(0,len(x)):
                    j=len(x)-k-1
                    filall.write("%g,%g\n" % (x[j],aball[i][j]-aball[0][j]))
            for i in range(1,pars.pignums+1):
                text=str(i)+p8+st+'[FWHM='+str(exciton[0].bandwidth)+']-'+directory+'-diff\n'+str(len(x))+'\n0\n'
                filall.write('CD-diff'+text)
                for k in range(0,len(x)):
                    j=len(x)-k-1
                    filall.write("%g,%g\n" % (x[j],cdall[i][j]-cdall[0][j]))
            filall.close()
        self.exciton = exciton
        self.ab = ab
        self.cd = cd
        self.x = x
        self.directory = directory
        self.pars = pars
        
    def get_average_ham(self,pars):
        count=0
        for ex in self.exciton:
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
        exa.bandwidth = math.sqrt(pars.bandwidth**2 + 90**2)
        #exa7=copy.deepcopy(exa)
        exa.getsticks()
        exa.getspectra()
        #exa7.getsticks()
        #exa7.getspectra()
        
        #absa=exa.abs #*0.55 + exa7.abs*0.45
        #cda=exa.CD #*0.55 + exa7.CD*0.45
        absa=exa.abs
        cda=exa.CD
        x = 1e7/exa.x
        
        fig, (ax1,ax2) = plt.subplots(2,1, sharex=True)
        ax1.plot(x, absa)
        ax2.plot(x,cda)
        
        ax1.set(xlabel='', ylabel='absorbance',title='Exciton of average' + self.directory)
        ax2.set(xlabel='wavenumers', ylabel='CD')
        ax1.grid()
        ax2.grid()
        fig.savefig(self.directory+'AverageHam-'+str(int(exa.bandwidth))+'.jpg')
        
        self.absa = absa
        self.cda = cda
        self.exa = exa


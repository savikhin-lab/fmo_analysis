import numpy as np
import os
import os.path
import scipy as sc
from scipy.spatial.transform import Rotation as Rotation
import distutils.dir_util
from ClassExciton_dual import Exciton
from ClassExciton_dual import Pigment
import math

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector
#https://github.com/nghiaho12/rigid_transform_3D/blob/master/rigid_transform_3D.py
# https://nghiaho.com/?page_id=671
def transform_3D(A, B):  # assumes that structures are centered!!!
    H = A @ np.transpose(B)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    # special reflection case
    if np.linalg.det(R) < 0:
        #print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
        R = Vt.T @ U.T
    #t = -R @ centroid_A + centroid_B
    return R

"""
s = Snapshots(directory)
    define directory with all structure files. File file.txt must contain 
    list of all structure files (in the same directory), one name per line
s.N - how many snapshots were loade
s.hams[i] - NxN matrix, hamiltonian for snapshot i
s.tdms[i] - Nx3 vectors, s.tdms[i][p] TDM for pigment [p] for snapshot [i] (size - 3, vector)
s.coords[i] - N vectors, coordinate for each pigment in snapshot i
s.snapshot_files[i] - from which file snapshot i was loaded (filename)
s.directory - from which directory (dir name)
s.save(ham,tdm,coord,['filename']) - save snapshot into file in the same format as snapshot files
ham,tдm,coord = s.load(filename) - loads from snapshot file (used in __init__)
s.center_structures() - moves xyz coordinates of each structure so that center of mass is at origin (0,0,0)
s.adjust_structures() - rotates all structures for best match between them
    It also makes s.aver_hams, s.aver_tdms, s.aver_coords that represent average snapshot parameters
    aver_hams are just average of all values, aver_coords is also average of all, 
    and for aver_tdms tdms are first normalized to unit length, then average is found
    and its length is set to average of lengths of all tdms
s.savestructures(subdir='rotated') - saves ALL structures into subdir of directory (makes it if needed)
    also saves average.csv for average structure
"""

class Snapshots:
    def __init__(self,directory,limit=0):
        self.directory = directory
        files = directory + '/files.txt' #list of names of data files (snapshots)
        filenames = []
        self.snapshot_files = []
        with open(files) as f:
            filenames = []
            for line in f: filenames.append(line.split()[0])        
        # now filenames is a list of all datafiles in directory (from files.txtx)
        # We will read all the data now
        hams = []       # will store all hamiltonians
        tdms = []    # transition dipole moments
        coords = []     # center coordinates
        
        # Load data from all files
        if limit<=0: limit = len(filenames)
        i = 0
        for nam in filenames:
            namh = directory + '/' + nam
            if os.path.isfile(namh) == False:
                print("Hamiltonian file does not exist, skip:")
                print(namh)
                continue
            if os.access(namh, os.R_OK) == False:
                print("Hamiltonian file is not accessible, skip:")
                print(namh)
                continue
            ham,tdm,coord = self.load(namh)
            hams.append(ham)
            tdms.append(tdm)
            coords.append(coord)
            self.snapshot_files.append(nam)
            i = i+1
            if i>=limit: break
        self.hams = np.array(hams)
        self.tdms = np.array(tdms)
        self.coords = np.array(coords)
        self.N = len(self.coords)
        self.has_averages = False
        return
    def load(self,filename):
        namh = filename
        with open(namh) as f:
            data = []
            for line in f: # read rest of lines
                data.append([float(x) for x in line.split()])
        # data will be 8x(8+6) matrix if everything is correct, check it:
        N = len(data)
        for m in range(0,N):
            if N != len(data[m])-6:
                print('!!! HAMILTONIAN MUST BE SQUARE + mu(3) + pos(3)!!!')
                print('File: ',namh)
                print('# of lines = ',len(ham))
                print('line ',m,' length = ',len(ham[m]))
                exit()
        #extract hamiltonian
        ham = np.zeros((N,N))
        tdm = np.zeros((N,3))
        coord = np.zeros((N,3))
        for m in range(0,N):
            for n in range(0,N):
                ham[m,n] = data[m][n] # the order m,n not important since syppetric, but in python we must invert it this way for general compatibility
            for n in range(0,3):
                tdm[m,n] = data[m][N+n]
                coord[m,n] = data[m][N+3+n]
        return ham,tdm,coord        
    
    def get_hams(self):
        return self.hams
    def get_tdms(self):
        return self.tdms
    def get_coords(self):
        return self.tdms
    def save(self,hama,tdma,coorda,filename='average.csv'):
        f = open(self.directory+'/'+filename,"w")
        for i in range(0,len(hama)):
            for j in range(0,len(hama)):
                f.write('%20.15g' % hama[j][i])
            for j in range(0,3):
                f.write('%15.5g' % tdma[i][j])
            for j in range(0,3):
                f.write('%15.5g' % coorda[i][j])
            f.write('\n')
        f.close()
    def center_structures(self):
        coords = []
        for i in range(0,self.N):
            self.coords[i] = self.coords[i]-np.average(self.coords[i],0)
        return
    
    def adjust_structures_SVD(self,max_iterations = 100,rmsd_cutoff = 1e-8,rotate_tdms=True,rotate_coords=True,verbose = True):
        self.center_structures()
        ### find rotation matrices for minimum rmsd and average
        average = np.average(self.coords,axis=0) #self.coords[0]  # use first structure as zero approximation and find rotations for all others
        tdms = np.array(self.tdms)    
        rmsd_old = 1e50
        old_coords = np.array(self.coords)
        for k in range(0,max_iterations):  #run maximum of 100 iterations
            rmsd = 0
            for i in range(0,self.N):
                R = transform_3D(self.coords[i].transpose(),average.transpose())
                self.coords[i] = self.coords[i] @ R.transpose()
                if rotate_tdms == True: tdms[i] = tdms[i] @ R.transpose() 
                rmsd += np.sum(np.square(average-self.coords[i])) 
            rmsd = math.sqrt(rmsd/self.N/len(self.coords[0]))
            if verbose: print(f'Iterration {k}: rmsd = {rmsd}')
            if abs(rmsd_old-rmsd)/rmsd < 1e-8: break
            average = np.average(self.coords,axis=0) # new average structure
            rmsd_old = rmsd       
        if verbose: print('Finished')
        
        ##### calculate average TDM: ONLY direction, so normalize first
        tdmaver = np.zeros((len(self.tdms[0]),3))
        tdmamp = np.zeros((len(self.tdms[0])))
        for j in range(0,len(self.coords[0])):
            for i in range(0,self.N):
                tdmaver[j] = tdmaver[j] + tdms[i][j]/np.linalg.norm(tdms[i][j]) # get average of directions
                tdmamp[j] += np.linalg.norm(tdms[i][j]) #calculate amplitude average separately
            tdmamp[j] /= self.N
            tdmaver[j] = tdmaver[j] / np.linalg.norm(tdmaver[j]) * tdmamp[j] #average directoion and average amplitude
        self.aver_tdms = tdmaver
        self.aver_hams = np.average(self.hams,axis=0)
        self.aver_coords = average
        self.has_averages = True
        if rotate_coords == False:
            self.coords = np.array(old_coords)
        return self.aver_hams, self.aver_tdms, self.aver_coords        
        
    def adjust_structures(self,max_iterations = 100,rmsd_cutoff = 1e-8,verbose = True):
        self.center_structures()
        ### find rotation matrices for minimum rmsd and average
        average = self.coords[0]  # use first structure as zero approximation and find rotations for all others
        tdms = np.array(self.tdms)    
        rmsd_old = 1e50
        for k in range(0,max_iterations):  #run maximum of 100 iterations
            rmsd = 0
            for i in range(0,self.N):
                rot1,rmsd1 = Rotation.align_vectors(average,self.coords[i]) # align vecors by rmsd, get rmsd
                self.coords[i] = rot1.apply(self.coords[i])  # rotate coordinates
                tdms[i] = rot1.apply(tdms[i])      # rotate tdms
                rmsd += rmsd1**2 
            rmsd = math.sqrt(rmsd/self.N/len(self.coords[0]))
            if verbose: print(f'Iterration {k}: rmsd = {rmsd}')
            if abs(rmsd_old-rmsd)/rmsd < 1e-8: break
            average = np.average(self.coords,axis=0) # new average structure
            rmsd_old = rmsd       
        if verbose: print('Finished')
        
        ##### calculate average TDM: ONLY direction, so normalize first
        tdmaver = np.zeros((len(self.tdms[0]),3))
        tdmamp = np.zeros((len(self.tdms[0])))
        for j in range(0,len(self.coords[0])):
            for i in range(0,self.N):
                tdmaver[j] = tdmaver[j] + tdms[i][j]/np.linalg.norm(tdms[i][j]) # get average of directions
                tdmamp[j] += np.linalg.norm(tdms[i][j]) #calculate amplitude average separately
            tdmamp[j] /= self.N
            tdmaver[j] = tdmaver[j] / np.linalg.norm(tdmaver[j]) * tdmamp[j] #average directoion and average amplitude
        self.aver_tdms = tdmaver
        self.aver_hams = np.average(self.hams,axis=0)
        self.aver_coords = average
        self.has_averages = True
        return self.aver_hams, self.aver_tdms, self.aver_coords
    
    def save_structures(self,subdir='rotated'):
         # output
        distutils.dir_util.mkpath(self.directory+'/'+subdir)
        for i in range(0,self.N):
            self.save(self.hams[i],self.tdms[i],self.coords[i],subdir+'/'+self.snapshot_files[i])
        if self.has_averages:
            self.save(self.aver_hams,self.aver_tdms,self.aver_coords,'rotated/average.csv')       
        return
    
    def ConstructExcitons(self,bandwidth=70,bandwidth_for_average = 170,xfrom=11790.,xto=13300,xstep=1,shift_diag = -2420):
        self.excitons = []
        for i in range(0,self.N):
            ex = Exciton()
            pig = []
            N=len(self.hams[i])
            for m in range(0,N):
                p = Pigment()
                p.mu = np.copy(self.tdms[i][m])
                p.coord =  np.copy(self.coords[i][m])
                p.mag = 0.
                pig.append(p)
            ex.setsystem(self.hams[i],pig)
            for k in range(0,ex.size): ex.ham[k][k] += shift_diag
            ex.bandwidth = bandwidth
            ex.setlimits(xfrom,xto,xstep)
            ex.getsticks()
            ex.getspectra()
            if i == 0:
                self.average_abs = np.copy(ex.abs)
                self.average_CD = np.copy(ex.CD)
            else:
                self.average_abs += ex.abs
                self.average_CD += ex.CD
            self.excitons.append(ex)
        self.average_abs /= self.N
        self.average_CD /= self.N
        # find spectra of average ham
        if self.has_averages == True:
            ex = Exciton()
            pig = []
            for m in range(0,len(self.aver_coords)):
                p = Pigment()
                p.mu = np.copy(self.aver_tdms[m])
                p.coord = np.copy(self.aver_coords[m])
                p.mag = 0.
                pig.append(p)
            ex.setsystem(self.aver_hams,pig)
            ex.bandwidth = bandwidth_for_average
            ex.setlimits(xfrom,xto,xstep)
            for k in range(0,ex.size): ex.ham[k][k] += shift_diag
            ex.getsticks()
            ex.getspectra()
            self.average_exciton = ex
            self.average_ham_abs = ex.abs
            self.average_ham_CD = ex.CD

            
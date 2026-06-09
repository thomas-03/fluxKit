import numpy as np
import os
import glob
import sys

# append the path of the
# parent directory
sys.path.append("..")
from montecarlo import joinlists
from montecarlo import make_spectrum
from montecarlo import spec2txt
from montecarlo import athena_mc as athenamc

class MCSpec:
    '''Class to store Monte Carlo spectra (including error bars).'''

    def __init__(self,directory,nproc,nfreq=None,emin=None,emax=None,listName=None,nout=1,outPath=None,overwrite=False):
        '''Initialize the MCSpec object by loading the spectra from the specified directory.'''
        if emin == None:
            inputFile = glob.glob(os.path.join(directory,'athinput.*'))

            if len(inputFile)>1:
                raise UserWarning('Multiple input files detected: '+inputFile+' \n Please specify which you want to use')
            
            nfreq,emin,emax = self.read_inputFile(inputFile[0])
        
        if listName is None:
            listFiles = glob.glob(os.path.join(directory,'*.list'))
            
            fileNameFormat = listFiles[0].split('/')[-1].split('.')
            outIndex = fileNameFormat.index('out1')
            listName = '.'.join(fileNameFormat[:outIndex+1])

        if outPath is None:
            outfile = os.path.join(directory,listName+".list")
            specFile = os.path.join(directory,listName+".spec")
            txtFile = os.path.join(directory,listName+".txt")
        else:
            if not os.path.isdir(outPath):
                os.makedirs(outPath)
            outfile = os.path.join(outPath,listName+".list")
            specFile = os.path.join(outPath,listName+".spec")
            txtFile = os.path.join(outPath,listName+".txt")
        if not os.path.isfile(outfile) or overwrite:
            if nout == 1:
                listArgs = {'basename': os.path.join(directory,listName), 'nproc': nproc, 'start': 0, 'end': 0, 'skip': True,'multi_out': False, 'outfile': outfile, 'skip': False, 'rm': False}
            else:
                listArgs = {'basename': os.path.join(directory,listName), 'nproc': nproc, 'start': 0, 'end': nout-1, 'skip': True,'multi_out': False, 'outfile': outfile, 'skip': False, 'rm': False}
            joinlists.main(**listArgs)

        if not os.path.isfile(specFile) or overwrite:
            #go from list file to spec file
            specArgs = {'infile':outfile,'nx':nfreq,'xmin':emin,'xmax':emax,'nmu':1,'mumin':0,'mumax':1,'phimin':0,'phimax':2*np.pi,'anglebin':'cartesian','xaxis':'ev','linearx':False,'calclum':False,'screen':'no_screen','outfile':None,'yerror':True}
            make_spectrum.main(**specArgs)
        
        spectrum = athenamc.read_spectrum(specFile)
        xfaces = spectrum['xfaces']
        x = 0.5*(xfaces[1:] + xfaces[:-1])
        nu = athenamc.get_frequency(spectrum['units'], xfaces)
        intensity = spectrum['intensity']
        errors = spectrum['errors']

        #average over azimuthal angle (phi)
        norm = 1./float(spectrum['nphi'])
        intensity = np.sum(intensity,axis=1)*norm
        errors = np.sqrt(np.sum((errors)**2,axis=1))*norm

        #sum over polar angle (theta)
        nmu = spectrum['nmu']
        #mumid = 0.5*(spectrum['mufaces'][1:]+spectrum['mufaces'][:-1])
        mumid=0.5*(spectrum['mufaces'][1:]+spectrum['mufaces'][:-1])
        intensity = np.tensordot(mumid,intensity,axes=[0,1])/nmu
        errors = np.sqrt(np.tensordot((mumid)**2,(errors)**2,axes=[0,1]))/nmu

        #when I sum over polar angle, MC falls below blacklight
        #when I just chose the 0 index, MC and blacklight match
        #mumid is 0.5
        #I had to get rid of a factor of 2 in blacklight reading. 
        #basically it sums over the front half only
            
        self.freq = nu
        self.lum = intensity[0,:]*nu
        self.lum_err = errors[0,:]*nu
        '''
        #now go from spec file to txt file
        if not os.path.isfile(txtFile) or overwrite:
            txtArgs = {'infile':specFile,'imu':'sum','iphi':'ave','xscale':'log','xmin':None,'xmax':None,'yscale':'log','ymin':None,'ymax':None,'xunit':'kev','yunit':'nulnu','ploterr':True,'outfile':None,'bbtemp':None,'bbnorm':None}
            self.freq,self.lum,self.lum_err = spec2txt.main(**txtArgs)
            self.freq = np.array(self.freq).reshape((nfreq,))
            self.lum = np.array(self.lum).reshape((nfreq,))
            self.lum_err = np.array(self.lum_err).reshape((nfreq,))
        else:
            mcResults = np.loadtxt(txtFile)
            self.freq = mcResults[:,0]
            self.lum = mcResults[:,1]
            self.lum_err = mcResults[:,2]
        '''
    

    def read_inputFile(self,inputFile):
        nfreq = 50 #default number of frequency bins to use if there are none \
        emin = None
        emax = None
        multRanges = False

        with open(inputFile,'r') as file:
            for line in file:
                line_txt = line.strip()
                if line_txt[:4]=='emin':
                    if emin == None:     
                        emin = float(line_txt.split('=')[-1])
                    #else:
                    #    emin = min(emin,float(line_txt.split('=')[-1]))
                elif line_txt[:4]=='emax':
                    if emax == None:
                        emax = float(line_txt.split('=')[-1])
                    #else:
                    #    emax = max(emax,float(line_txt.split('=')[-1]))
                elif line_txt[:5]=='nfreq':
                    if nfreq == None:
                        nfreq = int(line_txt.split('=')[-1])
                elif line_txt[:7]=='nf_scat':
                    nfreq = int(line_txt.split('=')[-1])
        return nfreq,emin,emax




#temp = MCSpec('/PellaShared/kcu8rf/spherical_compton_1e7/',32,overwrite=True)

#!/usr/bin/python
import os
import numpy as np


def run_mult_inclinations(blacklight_path,base_input_file,base_output_name,ninc,overwrite=False):
    i = np.linspace(0.0,180,ninc)
    outputs = []
    directory = base_output_name.split('/')[:-1]
    directory = "/".join(directory)
    directory += '/'
    print(directory)

    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(directory+'/'+f)]
    print(files)

    for j in range(ninc):
        output_name= base_output_name+"_i{:05.1f}.npz".format(i[j])
        outputs.append(output_name)
        if output_name in files and not overwrite:
            continue
        command = blacklight_path + " " + base_input_file + " --output_file="+output_name+" --camera_th={0}".format(i[j])
        os.system(command)
    return outputs



def produce_jobfile(blacklight_path,base_input_file,directory,jobfile='./jobfile'):
    '''Produce a job file that runs multiple snapshots through blacklight with the same base input file'''
    
    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(directory+'/'+f)]

    #assume standard Athena++ format 
    athfiles = [f for f in files if f[-6:]=='.athdf']
    outputs = [(f[:-6] + '.npz') for f in athfiles if f[-6:]=='.athdf'] 
    
    
    #if you have no Athena++ files, look for KHARMA ones
    kharfiles = []
    if len(athfiles)==0:
        kharfiles = [f for f in files if f[-5:]=='.phdf']
        outputs = [(f[:-5] + '.npz') for f in kharfiles]
        files = kharfiles
    

    with open(jobfile, "w") as f:
        for i in range(len(files)):
            command = blacklight_path + " " + base_input_file + " --output_file="+directory+'/'+outputs[i]+" --simulation_file="+directory+'/'+files[i]
            f.write(command)
            f.write('\n')
    return [directory+'/'+o for o in outputs]

def run_from_jobfile(jobfile):
    with open(jobfile,"+r") as file:
        for line in file:
            os.system(line)
    

def run_pylauncher(blacklight_path,base_input_file,directory,ncores):
    try:
        import pylauncher
    except ImportError:
        raise UserWarning('PyLauncher seems to be unavailable on your machine!')
    if ncores>32:
        raise UserWarning('Warning: blacklight does not have perfect scaling and running many jobs each with a few cores may be more efficient than using this many cores.')
    produce_jobfile(blacklight_path,base_input_file,directory)
    pylauncher.ClassicLauncher("commandlines",cores=ncores)

    files = os.listdir(directory)
    files = [f for f in files if os.path.isfile(directory+'/'+f)]
    files = [f for f in files if f[-6:]=='.npz']
    return files


    

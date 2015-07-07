import numpy as _np
import fileinput as _fileinput

def read_sp_files(files):
    """Read all *.singlepulse files in the current directory in a DM range.
        Return 5 arrays (properties of all single pulses):
                DM, sigma, time, sample, downfact."""
    finput = _fileinput.input(files)
    data = _np.loadtxt(finput,
                       dtype=_np.dtype([('dm', 'float32'),
                                        ('sigma','float32'),
                                        ('time','float32'),
                                        ('sample','uint32'),
                                        ('downfact','uint8')]))
    return _np.atleast_2d(data)

def read_tarfile(filenames, names, tar):
    """Read in the .singlepulse.tgz file instead of individual .singlepulse files.
        Return an array of (properties of all single pulses):
              DM, sigma, time, sample, downfact. 
        Input: filenames: names of all the singlepulse files.
               names: subset of filenames. Names of the singlepulse files to be 
               plotted in DM vs time.
               tar: tar file (.singlepulse.tgz)."""  
    members = []
    for name in names:
        if name in filenames:
            member = tar.getmember(name)
            members.append(member)
        else:
            pass
    fileinfo = []
    filearr = []
    for mem in members:
        file = tar.extractfile(mem)
        for line in file.readlines():
            fileinfo.append(line)
        filearr+=(fileinfo[1:])  #Removes the text labels ("DM", "sigma" etc) of the singlepulse properties. Only keeps the values. 
        fileinfo = []
    temp_list = []
    for i in range(len(filearr)):
        temp_line = filearr[i].split()
        temp_list.append(temp_line)
    main_array = _np.asarray(temp_list)
    main_array = _np.split(main_array, 5, axis=1)
    main_array[0] = main_array[0].astype(_np.float16)
    main_array[1] = main_array[1].astype(_np.float16)
    main_array[2] = main_array[2].astype(_np.float16)
    main_array[3] = main_array[3].astype(_np.int)
    main_array[4] = main_array[4].astype(_np.int)
    return main_array
def gen_arrays(dm, threshold, sp_files, tar):    
    """
    Extract dms, times and signal to noise from each singlepulse file as 1D arrays.
    Input: 
           dm: The dm array of the main pulse. Used to decide the DM range in the DM vs time plot and pick out singlepulse files with those DMs.
           threshold: Min signal to noise of the single pulse event that is plotted.
           sp_files: all the .singlepulse file names.
           tar: Instead of the providing individual singlepulse files, you can provide the .singlepulse.tgz tarball.
    Output:
           Arrays: dms, times, sigmas of the singlepulse events and an array of dm_vs_times file names.
           
    Options: Either a tarball of singlepulse files or individual singlepulse files can be supplied.
             Faster when individual singlepulse files are supplied.   
    """
    max_dm = _np.ceil(_np.max(dm)).astype('int')
    min_dm = _np.min(dm).astype('int')
    diff_dm = max_dm-min_dm
    ddm = min_dm-diff_dm
    if (ddm <= 0):
        ddm = 0
    dmss = _np.zeros((1,)).astype('float32')
    timess = _np.zeros((1,)).astype('float32')
    sigmass = _np.zeros((1,)).astype('float32')
    ind = []
    dm_time_files = []
    for i in range(ddm,(max_dm+diff_dm)):
        """after DM of 1826 the dm step size is >=1, therefore we need to pick the correct DMs."""
        if (i >= 1826) and (i < 3266):
            if int(i)%2 == 1:
                i = i+1
            try:
                singlepulsefiles = [sp_files[sp_file] for sp_file in range(len(sp_files)) if ('DM'+str(i)+'.') in sp_files[sp_file]]
                dm_time_files += singlepulsefiles
                if tar is not None:
                    data = read_tarfile(sp_files, singlepulsefiles, tar)
                else:
                    data = read_sp_files(singlepulsefiles)[0]
            except:
                pass
        elif (i >= 3266) and (i < 5546):
            if int(i)%3 == 0:
                i = i+2
            if int(i)%3 == 1:
                i = i+1
            try:
                singlepulsefiles = [sp_files[sp_file] for sp_file in range(len(sp_files)) if ('DM'+str(i)+'.') in sp_files[sp_file]]
                dm_time_files += singlepulsefiles
                if tar is not None:
                    data = read_tarfile(sp_files, singlepulsefiles, tar)
                else:
                    data = read_sp_files(singlepulsefiles)[0]
            except:
                pass
        elif i>=5546:
            if int(i)%5 == 2:
                i = i+4
            if int(i)%5 == 3:
                i = i+3
            if int(i)%5 == 4:
                i = i+2
            if int(i)%5 == 0:
                i = i+1
            try:
                singlepulsefiles = [sp_files[sp_file] for sp_file in range(len(sp_files)) if ('DM'+str(i)+'.') in sp_files[sp_file]]
                dm_time_files += singlepulsefiles
                if tar is not None:
                    data = read_tarfile(sp_files, singlepulsefiles, tar)
                else:
                    data = read_sp_files(singlepulsefiles)[0]
            except:
                pass
        else:    
            try:
                singlepulsefiles = [sp_files[sp_file] for sp_file in range(len(sp_files)) if ('DM'+str(i)+'.') in sp_files[sp_file]]
                dm_time_files += singlepulsefiles
                if tar is not None:
                    data = read_tarfile(sp_files, singlepulsefiles, tar)
                else:
                    data = read_sp_files(singlepulsefiles)[0]
            except:
                pass
        if tar is not None:
            dms = _np.reshape(data[0],(len(data[0]),))
            times = _np.reshape(data[2],(len(data[1]),))
            sigmas = _np.reshape(data[1],(len(data[2]),))
        else:
            dms = data['dm']
            times = data['time']
            sigmas = data['sigma']
        dms = _np.concatenate((dmss, dms), axis = 0)
        dmss = dms
        times = _np.concatenate((timess, times), axis = 0)
        timess = times
        sigmas = _np.concatenate((sigmass, sigmas), axis = 0)
        sigmass = sigmas
    dms = _np.delete(dms, (0), axis = 0)
    times = _np.delete(times, (0), axis = 0)
    sigmas = _np.delete(sigmas, (0), axis = 0)
    return dms, times, sigmas, dm_time_files

def read_spd(spd_file, tar = None):
    """ 
       Reads in all the .spd and the .singlepulse.tgz info that can reproduce the sp plots.
       Inputs: spd_file: .spd file
               .singlepulse.tgz: if not supplied, it will only output .spd info. 
                                 Default: not supplied. 
       Output: An object that has all the relevant information to remake the plot. 
    """
    sp = spd(spd_file)
    if tar is not None:
        dmVt_dms, dmVt_times, dmVt_sigmas, dmVt_files = gen_arrays(sp.dmVt_this_dms, sp.spfiles, tar, threshold = 5)
        sp.dmVt_dms = dmVt_dms
        sp.dmVt_times = dmVt_times
        sp.dmVt_sigmas = dmVt_sigmas
        return sp
    else:
        return sp 



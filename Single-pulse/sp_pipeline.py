#!/usr/bin/env python

"""
waterfaller.py

Make waterfall plots to show frequency sweep of a single pulse.
Reads SIGPROC filterbank format.

Patrick Lazarus - Aug. 19, 2011
"""

import sys
import copy
from time import strftime
import infodata
from subprocess import Popen, PIPE

import numpy as np
import optparse
import waterfaller
import sp_pgplot 
import bary_and_topo
import psr_utils
import rfifind

from pypulsar.formats import psrfits
from pypulsar.formats import filterbank
from pypulsar.formats import spectra

DEBUG = True
def print_debug(msg):
    if DEBUG:
        print msg
def get_textfile():
    """ Read in the groups.txt file.
    Contains information about the DM, time, box car width, signal to noise, sample number and rank    of groups. 
    """
    return  np.loadtxt(options.txtfile,dtype = 'str',delimiter='\n')
    #Get The parameters  
def group_info(rank):
    """
    Extracts out relevant information from the groups.txt file as strings. 
    """
    file = get_textfile()
    lis=np.where(file == '\tRank:             %i.000000'%rank)[0]#Checks for this contidion and gives its indices where true.
    # Extract the Max_ sigma value for the required parameters
    parameters=[]
    temp_lines = []
    for i in range(len(lis)):
        temp_line = file[lis[i]-1]
        temp_list = temp_line.split()
        max_sigma = temp_list[2]
        max_sigma = float(max_sigma)
        max_sigma = '%.2f'%max_sigma
        # Extract the number of pulses for this group
        temp_line = file[lis[i]-6]
        temp_list = temp_line.split()
        number_of_pulses = int(temp_list[2])
        # Slice off a mini array to get the parameters from
        temp_lines = file[(lis[i]+1):(lis[i]+number_of_pulses+1)]
        # Get the parameters as strings containing the max_sigma
        params = temp_lines[np.array([max_sigma in line for line in temp_lines])]
        parameters.append(params)
    return temp_lines, parameters 
def split_parameters(rank):
    """
    Splits the string into individual parameters and converts them into floats/int. 
    """
    temp_values = group_info(rank)[1]
    final_parameters=[]
    for i in range(len(temp_values)):
        x = temp_values[i]
    # If there is a degeneracy in max_sigma values, Picks the first one.(Can be updated to get the best pick) 
        correct_list = x[0]
        correct_values = correct_list.split()
        correct_values[0] = float(correct_values[0])
        correct_values[1] = float(correct_values[1])
        correct_values[1] = float('%.2f'%correct_values[1])
        correct_values[2] = float(correct_values[2])
        correct_values[3] = int(correct_values[3])
        correct_values[4] = int(correct_values[4])
        final_parameters.append(correct_values)
    return final_parameters

def topo_timeshift(bary_start_time, time_shift, topo):
    ind = np.where(topo == float(int(bary_start_time)/10*10))[0]
    return time_shift[ind]

def maskdata(data, start_bin, nbinsextra):
    """
    Performs the masking on the raw data using the boolean array from get_mask.
    Inputs:
        data: raw data (psrfits object) 
        start_bin: the sample number where we want the waterfall plot window to start.
        nbinsextra: number of bins in the waterfall plot
    Output:
        data: 2D array after masking. 
    """
    print 'masking ...'
    maskfile = options.maskfile
    if maskfile is not None:
        rfimask = rfifind.rfifind(maskfile)
        mask = waterfaller.get_mask(rfimask, start_bin, nbinsextra)
        # Mask data
        data = data.masked(mask, maskval='median-mid80')
    return data
        
def main():
    file = get_textfile()
    print_debug("Begining waterfaller... "+strftime("%Y-%m-%d %H:%M:%S"))
    Detrendlen = 50
    fn = args[0]
    if not fn.endswith("fits"):
        raise ValueError("The first file must be a psrFits file! ") 
    basename = fn[:-5]
    filetype = "psrfits"
    inffile = options.infile
    topo, bary = bary_and_topo.bary_to_topo(inffile)
    time_shift = bary-topo
    inf = infodata.infodata(inffile)
    RA = inf.RA
    dec = inf.DEC
    MJD = inf.epoch
    mjd = Popen(["mjd2cal", "%f"%MJD], stdout=PIPE, stderr=PIPE)
    date, err = mjd.communicate()
    date = date.split()[2:5]
    telescope = inf.telescope
    N = inf.N
    Total_observed_time = inf.dt *N
    print 'getting file'
    rawdatafile = psrfits.PsrfitsFile(fn)
    bin_shift = np.round(time_shift/rawdatafile.tsamp).astype('int')
    for group in [6, 5, 4, 3, 2]:
        rank = group+1
        if file[group] != "Number of rank %i groups: 0 "%rank:
            print file[group]
            values = split_parameters(rank)
            DM = 10000.0
            dms = []
            time = []
            sigmas = []
            lis = np.where(file == '\tRank:             %i.000000'%rank)[0]
            for ii in range(len(values)):
                j = ii+1
                subdm = dm = sweep_dm= values[ii][0]
                integrate_dm = None
                sigma = values[ii][1]
                sweep_posn = 0.0
                topo_start_time = values[ii][2] - topo_timeshift(values[ii][2], time_shift, topo)[0]
                sample_number = values[ii][3]
                width_bins = values[ii][4]
                binratio = 50
                scaleindep = False
                zerodm = None
                downsamp = np.round((values[ii][2]/sample_number/6.54761904761905e-05)).astype('int')
                duration = binratio * width_bins * rawdatafile.tsamp * downsamp
                start = topo_start_time - (0.25 * duration)
                if (start<0.0):
                    start = 0.0
                pulse_width = width_bins*downsamp*6.54761904761905e-05
                if sigma <= 10:
                    nsub = 32
                elif sigma >= 10 and sigma < 15:
                    nsub = 64
                else:
                    nsub = 96
                nbins = np.round(duration/rawdatafile.tsamp).astype('int')
                start_bin = np.round(start/rawdatafile.tsamp).astype('int')
                dmfac = 4.15e3 * np.abs(1./rawdatafile.frequencies[0]**2 - 1./rawdatafile.frequencies[-1]**2)
                nbinsextra = np.round((duration + dmfac * dm)/rawdatafile.tsamp).astype('int')
                if (start_bin+nbinsextra) > N-1:
                    nbinsextra = N-1-start_bin
                dat = rawdatafile.get_spectra(start_bin, nbinsextra)
                masked_dat = maskdata(dat, start_bin, nbinsextra)
                zerodm_masked_dat = copy.copy(masked_dat)

                #make an array to store header information for the .npz files
                temp_filename = basename+"_DM%.1f_%.1fs_rank_%i"%(subdm, topo_start_time, rank)
                text_array = np.array([fn, 'Arecibo', RA, dec, MJD, rank, nsub, nbins, subdm, sigma, sample_number, duration, width_bins, pulse_width, rawdatafile.tsamp, Total_observed_time, topo_start_time])
                # Plotting Dedispersed waterfall plot - zerodm - OFF
                data, bins = waterfaller.waterfall(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, masked_dat)
                Data = np.array(data.data)
                ragfac = float(nbins)/bins
                dmrange, trange = Data.shape
                nbinlim = np.int(trange * ragfac)
                array = Data[..., :nbinlim]
                array = array[::-1]
                Data_dedisp_nozerodm = array
                # Add additional information to the header information array
                text_array = np.append(text_array,data.starttime)
                text_array = np.append(text_array,data.dt)
                text_array = np.append(text_array,data.numspectra)
                text_array = np.append(text_array,data.freqs.min())
                text_array = np.append(text_array,data.freqs.max())
                print "plotting waterfall plots and dedispersed time series..." 
                sp_pgplot.ppgplot.pgopen(temp_filename+'.ps/VPS')
                sp_pgplot.ppgplot.pgpap(10.25, 8.5/11.0)

                sp_pgplot.ppgplot.pgsvp(0.07, 0.40, 0.50, 0.80)
                sp_pgplot.ppgplot.pgswin(data.starttime - start, data.starttime-start+nbinlim*data.dt, data.freqs.min(), data.freqs.max())
                sp_pgplot.ppgplot.pgsch(0.8)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgbox("BCST", 0, 0, "BCNST", 0, 0)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgmtxt('L', 1.8, 0.5, 0.5, "Observing Frequency (MHz)")
                sp_pgplot.ppgplot.pgmtxt('R', 1.8, 0.5, 0.5, "Zero-dm filtering - Off")
                sp_pgplot.plot_waterfall(array,rangex = [data.starttime-start, data.starttime-start+nbinlim*data.dt], rangey = [data.freqs.min(), data.freqs.max()], image = 'apjgrey')

               #### Plot Dedispersed Time series - Zerodm filter - Off
                integrate_dm = subdm
                Dedisp_ts = Data.sum(axis=0)
                times = (np.arange(data.numspectra)*data.dt+data.starttime-start)[..., :nbinlim]
                sp_pgplot.ppgplot.pgsvp(0.07, 0.40, 0.80, 0.90)
                sp_pgplot.ppgplot.pgswin(data.starttime - start, data.starttime-start+ nbinlim*data.dt, np.min(Dedisp_ts), 1.05*np.max(Dedisp_ts))
                sp_pgplot.ppgplot.pgsch(0.8)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgbox("BC", 0, 0, "BC", 0, 0)
                sp_pgplot.ppgplot.pgsci(1)
                sp_pgplot.ppgplot.pgline(times,Dedisp_ts)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgsci(1)
                errx1 = np.array([0.60 * (data.starttime-start+nbinlim*data.dt)])
                erry1 = np.array([0.60 * np.max(Dedisp_ts)])
                erry2 = np.array([np.std(Dedisp_ts)])
                errx2 = np.array([pulse_width])
                sp_pgplot.ppgplot.pgerrb(5, errx1, erry1, errx2, 1.0)
                sp_pgplot.ppgplot.pgpt(errx1, erry1, -1)

                #### Dedispersed with zerodm
                zerodm = True
                data, bins = waterfaller.waterfall(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, zerodm_masked_dat)
                Data = np.array(data.data)
                ragfac = float(nbins)/bins
                dmrange, trange = Data.shape
                nbinlim = np.int(trange * ragfac)
                array = Data[..., :nbinlim]
                array = array[::-1]
                Data_dedisp_zerodm = array
                sp_pgplot.ppgplot.pgsvp(0.07, 0.40, 0.1, 0.40)
                sp_pgplot.ppgplot.pgswin(data.starttime-start , data.starttime-start+ nbinlim*data.dt, data.freqs.min(), data.freqs.max())
                sp_pgplot.ppgplot.pgsch(0.8)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgbox("BCNST", 0, 0, "BCNST", 0, 0)
                sp_pgplot.ppgplot.pgmtxt('B', 2.5, 0.5, 0.5, "Time - %.2f s"%data.starttime)
                sp_pgplot.ppgplot.pgmtxt('L', 1.8, 0.5, 0.5, "Observing Frequency (MHz)")
                sp_pgplot.ppgplot.pgmtxt('R', 1.8, 0.5, 0.5, "Zero-dm filtering - On")
                sp_pgplot.plot_waterfall(array,rangex = [data.starttime-start, data.starttime-start+ nbinlim*data.dt],rangey = [data.freqs.min(), data.freqs.max()],image = 'apjgrey')
                   #### Plot Dedispersed Time series - Zerodm filter - On
                times = (np.arange(data.numspectra)*data.dt+data.starttime-start)[..., :nbinlim]
                dedisp_ts = Data.sum(axis=0)
                sp_pgplot.ppgplot.pgsvp(0.07, 0.40, 0.40, 0.50)
                sp_pgplot.ppgplot.pgswin(data.starttime - start, data.starttime-start+ nbinlim*data.dt, np.min(dedisp_ts), 1.05*np.max(dedisp_ts))
                sp_pgplot.ppgplot.pgsch(0.8)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgbox("BC", 0, 0, "BC", 0, 0)
                sp_pgplot.ppgplot.pgsci(1)
                sp_pgplot.ppgplot.pgline(times,dedisp_ts)
                errx1 = np.array([0.60 * (data.starttime-start+nbinlim*data.dt)])
                erry1 = np.array([0.60 * np.max(dedisp_ts)])
                erry2 = np.array([np.std(dedisp_ts)])
                errx2 = np.array([pulse_width])
                sp_pgplot.ppgplot.pgerrb(5, errx1, erry1, errx2, 1.0)
                sp_pgplot.ppgplot.pgpt(errx1, erry1, -1)
                integrate_dm = None

                #### Figure texts 
                sp_pgplot.ppgplot.pgsvp(0.745, 0.97, 0.64, 0.909)
                sp_pgplot.ppgplot.pgsch(0.85)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgmtxt('T', -1.1, 0.01, 0.0, "RA: %s" %RA)
                sp_pgplot.ppgplot.pgmtxt('T', -2.6, 0.01, 0.0, "DEC: %s" %dec)
                sp_pgplot.ppgplot.pgmtxt('T', -4.1, 0.01, 0.0, "MJD: %f" %MJD)
                sp_pgplot.ppgplot.pgmtxt('T', -5.6, 0.01, 0.0, "Observation date: %s %s %s" %(date[0], date[1], date[2]))
                sp_pgplot.ppgplot.pgmtxt('T', -7.1, 0.01, 0.0, "Telescope: %s" %telescope)
                sp_pgplot.ppgplot.pgmtxt('T', -8.6, 0.01, 0.0, "DM: %.2f pc cm\u-3\d" %dm)
                sp_pgplot.ppgplot.pgmtxt('T', -10.1, 0.01, 0.0, "S/N\dMAX\u: %.2f" %sigma)
                sp_pgplot.ppgplot.pgmtxt('T', -11.6, 0.01, 0.0, "Number of samples: %i" %nbins)
                sp_pgplot.ppgplot.pgmtxt('T', -13.1, 0.01, 0.0, "Number of subbands: %i" %nsub)
                sp_pgplot.ppgplot.pgmtxt('T', -14.6, 0.01, 0.0, "Pulse width: %.2f ms" %(pulse_width*1e3))
                sp_pgplot.ppgplot.pgmtxt('T', -16.1, 0.01, 0.0, "Sampling time: %.2f \gms" %(rawdatafile.tsamp*1e6))
                sp_pgplot.ppgplot.pgsvp(0.07, 0.7, 0.01, 0.05)
                sp_pgplot.ppgplot.pgmtxt('T', -2.1, 0.01, 0.0, "%s"%fn)

                ####Sweeped without zerodm
                start = start + (0.25*duration)
                start_bin = np.round(start/rawdatafile.tsamp).astype('int')
                sweep_duration = 4.15e3 * np.abs(1./rawdatafile.frequencies[0]**2-1./rawdatafile.frequencies[-1]**2)*sweep_dm
                nbins = np.round(sweep_duration/(rawdatafile.tsamp)).astype('int')
                if ((nbins+start_bin)> (N-1)):
                    nbins = N-1-start_bin
                dat = rawdatafile.get_spectra(start_bin, nbins)
                masked_dat = maskdata(dat, start_bin, nbins)
                zerodm = None
                dm = None
                zerodm_masked_dat = copy.copy(masked_dat)
                data, bins = waterfaller.waterfall(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, masked_dat)
                text_array = np.append(text_array, sweep_duration)
                text_array = np.append(text_array, data.starttime)
                Data = np.array(data.data)[..., :nbins]
                array = Data
                array = array[::-1]
                Data_nozerodm = array
                sp_pgplot.ppgplot.pgsvp(0.20, 0.40, 0.50, 0.70)
                sp_pgplot.ppgplot.pgswin(data.starttime, data.starttime+sweep_duration, data.freqs.min(), data.freqs.max())
                sp_pgplot.ppgplot.pgsch(0.8)
                sp_pgplot.ppgplot.pgslw(4)
                sp_pgplot.ppgplot.pgbox("BCST", 0, 0, "BCST", 0, 0)
                sp_pgplot.ppgplot.pgsch(3)
                # Construct the image
                sp_pgplot.plot_waterfall(array,rangex = [data.starttime, data.starttime+sweep_duration],rangey = [data.freqs.min(), data.freqs.max()],image = 'apjgrey')
                if sweep_dm is not None:
                    ddm = sweep_dm-data.dm
                    delays = psr_utils.delay_from_DM(ddm, data.freqs)
                    delays -= delays.min()
                    delays_nozerodm = delays
                    freqs_nozerodm = data.freqs
                    sp_pgplot.ppgplot.pgslw(5)
                    sweepstart = data.starttime- 0.3*sweep_duration
                    sp_pgplot.ppgplot.pgsci(0)
                    sp_pgplot.ppgplot.pgline(delays+sweepstart, data.freqs)
                    sp_pgplot.ppgplot.pgsci(1)
                    sp_pgplot.ppgplot.pgslw(3)
                # Sweeped with zerodm-on 
                zerodm = True
                downsamp_temp = 1
                data, bins = waterfaller.waterfall(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp_temp, scaleindep, width_bins, rawdatafile, binratio, masked_dat)
                Data = np.array(data.data)[..., :nbins]
                array = Data
                array = array[::-1]
                Data_zerodm = array
                sp_pgplot.ppgplot.pgsvp(0.20, 0.40, 0.1, 0.3)
                sp_pgplot.ppgplot.pgswin(data.starttime, data.starttime+sweep_duration, data.freqs.min(), data.freqs.max())
                sp_pgplot.ppgplot.pgsch(0.8)
                sp_pgplot.ppgplot.pgslw(4)
                sp_pgplot.ppgplot.pgbox("BCST", 0, 0, "BCST", 0, 0)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.plot_waterfall(array,rangex = [data.starttime, data.starttime+sweep_duration],rangey = [data.freqs.min(), data.freqs.max()],image = 'apjgrey')
                if sweep_dm is not None:
                    ddm = sweep_dm-data.dm
                    delays = psr_utils.delay_from_DM(ddm, data.freqs)
                    delays -= delays.min()
                    sp_pgplot.ppgplot.pgslw(5)
                    sweepstart = data.starttime-0.3*sweep_duration
                    sp_pgplot.ppgplot.pgsci(0)
                    sp_pgplot.ppgplot.pgline(delays+sweepstart, data.freqs)
                    sp_pgplot.ppgplot.pgsci(1)
                    sp_pgplot.ppgplot.pgslw(3)
                #### Plotting DM vs SNR
                temp_line = file[lis[ii]-6]
                temp_list = temp_line.split()
                npulses = int(temp_list[2])
                temp_lines = file[(lis[ii]+3):(lis[ii]+npulses+1)]
                arr = np.split(temp_lines, len(temp_lines))
                dm_list = []
                time_list = []
                for i in range(len(arr)):
                    dm_val= float(arr[i][0].split()[0])
                    time_val = float(arr[i][0].split()[2])
                    dm_list.append(dm_val)
                    time_list.append(time_val)
                arr_2 = np.array([arr[i][0].split() for i in range(len(arr))], dtype = np.float32)
                dm_arr = np.array([arr_2[i][0] for i in range(len(arr))], dtype = np.float32)
                sigma_arr = np.array([arr_2[i][1] for i in range(len(arr))], dtype = np.float32)
                time_arr = np.array([arr_2[i][2] for i in range(len(arr))], dtype = np.float32)
                print "plotting S/N vs DM ..." 
                sp_pgplot.ppgplot.pgsvp(0.48, 0.73, 0.65, 0.90)
                sp_pgplot.ppgplot.pgswin(np.min(dm_arr), np.max(dm_arr), 0.95*np.min(sigma_arr), 1.05*np.max(sigma_arr))
                sp_pgplot.ppgplot.pgsch(0.8)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgbox("BCNST", 0, 0, "BCNST", 0, 0)
                sp_pgplot.ppgplot.pgslw(3)
                sp_pgplot.ppgplot.pgmtxt('B', 2.5, 0.5, 0.5, "DM (pc cm\u-3\d)")
                sp_pgplot.ppgplot.pgmtxt('L', 1.8, 0.5, 0.5, "Signal-to-noise")
                sp_pgplot.ppgplot.pgpt(dm_arr, sigma_arr, 20)
                #### Plotting DM vs Time
                if subdm <10.00:
                    threshold = 6.00
                else:
                    threshold = 5.50
                sp_files = args[1:]
                if (np.abs(subdm - DM) > 15):
                    dms, time, sigmas, singlepulse_files = sp_pgplot.gen_arrays(dm_arr, threshold, sp_files)
                    DM = subdm
                dm_range = dms
                dms = dm_range
                time_range = time
                times = time_range
                sigma_range = sigmas
                sigmas = sigma_range
                print "plotting DM vs Time ..."
                sp_pgplot.dm_time_plot(dm_range, time_range, sigma_range, dm_list, sigma_arr, time_list, Total_observed_time)
                with open(temp_filename+".spd", 'wb') as f:
                    np.savez_compressed(f, Data_dedisp_nozerodm = Data_dedisp_nozerodm.astype(np.float16), Data_dedisp_zerodm = Data_dedisp_zerodm.astype(np.float16), Data_nozerodm = Data_nozerodm.astype(np.float16), delays_nozerodm = delays_nozerodm, freqs_nozerodm = freqs_nozerodm, Data_zerodm = Data_zerodm.astype(np.float16), dm_arr= map(np.float16, dm_arr), sigma_arr = map(np.float16, sigma_arr), dm_list= map(np.float16, dm_list), time_list = map(np.float16, time_list), text_array = text_array, singlepulse_files = singlepulse_files)
                sp_pgplot.ppgplot.pgiden()
                sp_pgplot.ppgplot.pgclos()
                print_debug("Finished plot %i " %j+strftime("%Y-%m-%d %H:%M:%S"))
        print_debug("Finished group %i... "%rank+strftime("%Y-%m-%d %H:%M:%S"))
    print_debug("Finished running waterfaller... "+strftime("%Y-%m-%d %H:%M:%S"))


if __name__=='__main__':
    parser = optparse.OptionParser(prog="sp_pipeline..py", \
                        version=" Chitrang Patel (May. 12, 2015)", \
                        usage="%prog INFILE(PsrFits FILE, SINGLEPULSE FILES)", \
                        description="Create single pulse plots to show the " \
                                    "frequency sweeps of a single pulse,  " \
                    "DM vs time, and SNR vs DM,"\
                                    "in psrFits data.")
    parser.add_option('--infile', dest='infile', type='string', \
                        help="Give a .inf file to read the appropriate header information.")
    parser.add_option('--groupsfile', dest='txtfile', type='string', \
                        help="Give the groups.txt file to read in the groups information.") 
    parser.add_option('--mask', dest='maskfile', type='string', \
                        help="Mask file produced by rfifind. (Default: No Mask).", \
                        default=None)
    options, args = parser.parse_args()
    if not hasattr(options, 'infile'):
        raise ValueError("A .inf file must be given on the command line! ") 
    if not hasattr(options, 'txtfile'):
        raise ValueError("The groups.txt file must be given on the command line! ") 
    main()

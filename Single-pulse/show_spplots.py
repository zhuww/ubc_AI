#! /usr/bin/env python

import sys
import os
import numpy as np
import optparse
import sp_pgplot
from subprocess import Popen, PIPE


def main():
    temp_file = args[0]
    if not temp_file.endswith(".spd"):
	raise ValueError("The first file must be a .spd file")
    npzfile = np.load(temp_file)
    
    ##### Read in the header information and other required variables for the plots. ######
    text_array = npzfile['text_array']
    
    fn = text_array[0]
    telescope = text_array[1]
    RA = text_array[2]
    dec = text_array[3]
    MJD = float(text_array[4])
    mjd = Popen(["mjd2cal", "%f"%MJD], stdout=PIPE, stderr=PIPE)
    date, err = mjd.communicate()
    date = date.split()[2:5]
    rank = int(text_array[5])
    nsub = int(text_array[6])
    nbins = int(text_array[7])
    subdm = dm = sweep_dm = float(text_array[8])
    sigma = float(text_array[9])
    sample_number = int(text_array[10])
    duration = float(text_array[11])
    width_bins = int(text_array[12])
    pulse_width = float(text_array[13])
    tsamp = float(text_array[14])
    Total_observed_time = float(text_array[15])
    start = float(text_array[16])
    start = start - 0.25*duration
    datastart = float(text_array[17])
    datasamp = float(text_array[18])
    datanumspectra = float(text_array[19])
    min_freq = float(text_array[20])
    max_freq = float(text_array[21])
    sweep_duration = float(text_array[22])
    sweeped_start = float(text_array[23])
    if options.xwin:
        pgplot_device = "/XWIN"
    else:
        pgplot_device = ""
    # Dedispersed waterfall plot - zerodm - OFF
    array = npzfile['Data_dedisp_nozerodm'].astype(np.float64)
    print array.shape
    if pgplot_device:
        sp_pgplot.ppgplot.pgopen(pgplot_device)
    else:
        sp_pgplot.ppgplot.pgopen(fn[:-5]+'_DM%.1f_%.1fs_rank_%i.spd.ps/VPS'%(subdm, (start+0.25*duration), rank))
    
    sp_pgplot.ppgplot.pgpap(10.25, 8.5/11.0)
    sp_pgplot.ppgplot.pgsvp(0.07, 0.40, 0.50, 0.80)
    sp_pgplot.ppgplot.pgswin(datastart - start, datastart -start+datanumspectra*datasamp, min_freq, max_freq)
    sp_pgplot.ppgplot.pgsch(0.8)
    sp_pgplot.ppgplot.pgslw(3)
    sp_pgplot.ppgplot.pgbox("BCST", 0, 0, "BCNST", 0, 0)
    sp_pgplot.ppgplot.pgslw(3)
    sp_pgplot.ppgplot.pgmtxt('L', 1.8, 0.5, 0.5, "Observing Frequency (MHz)")
    sp_pgplot.ppgplot.pgmtxt('R', 1.8, 0.5, 0.5, "Zero-dm filtering - Off")
    sp_pgplot.plot_waterfall(array,rangex = [datastart-start, datastart-start+datanumspectra*datasamp], rangey = [min_freq, max_freq], image = 'apjgrey')
    
     #### Plot Dedispersed Time series - Zerodm filter - Off
    Dedisp_ts = array[::-1].sum(axis = 0)
    times = np.arange(datanumspectra)*datasamp
    sp_pgplot.ppgplot.pgsvp(0.07, 0.40, 0.80, 0.90)
    sp_pgplot.ppgplot.pgswin(datastart - start, datastart-start+duration, np.min(Dedisp_ts), 1.05*np.max(Dedisp_ts))
    sp_pgplot.ppgplot.pgsch(0.8)
    sp_pgplot.ppgplot.pgslw(3)
    sp_pgplot.ppgplot.pgbox("BC", 0, 0, "BC", 0, 0)
    sp_pgplot.ppgplot.pgsci(1)
    sp_pgplot.ppgplot.pgline(times,Dedisp_ts)
    sp_pgplot.ppgplot.pgslw(3)
    sp_pgplot.ppgplot.pgsci(1)
    
    errx1 = np.array([0.60 * (datastart-start+duration)])
    erry1 = np.array([0.60 * np.max(Dedisp_ts)])
    erry2 = np.array([np.std(Dedisp_ts)])
    errx2 = np.array([pulse_width])
    sp_pgplot.ppgplot.pgerrb(5, errx1, erry1, errx2, 1.0)
    sp_pgplot.ppgplot.pgpt(errx1, erry1, -1)
    
    #Dedispersed waterfall plot - Zerodm ON
    sp_pgplot.ppgplot.pgsvp(0.07, 0.40, 0.1, 0.40)
    sp_pgplot.ppgplot.pgswin(datastart-start , datastart-start+datanumspectra*datasamp, min_freq, max_freq)
    sp_pgplot.ppgplot.pgsch(0.8)
    sp_pgplot.ppgplot.pgslw(3)
    sp_pgplot.ppgplot.pgbox("BCNST", 0, 0, "BCNST", 0, 0)
    sp_pgplot.ppgplot.pgmtxt('B', 2.5, 0.5, 0.5, "Time - %.2f s"%datastart)
    sp_pgplot.ppgplot.pgmtxt('L', 1.8, 0.5, 0.5, "Observing Frequency (MHz)")
    sp_pgplot.ppgplot.pgmtxt('R', 1.8, 0.5, 0.5, "Zero-dm filtering - On")
    array = npzfile['Data_dedisp_zerodm'].astype(np.float64)
    sp_pgplot.plot_waterfall(array,rangex = [datastart-start, datastart-start+datanumspectra*datasamp],rangey = [min_freq, max_freq],image = 'apjgrey')
    #### Plot Dedispersed Time series - Zerodm filter - On
    dedisp_ts = array[::-1].sum(axis = 0)
    times = np.arange(datanumspectra)*datasamp
    sp_pgplot.ppgplot.pgsvp(0.07, 0.40, 0.40, 0.50)
    sp_pgplot.ppgplot.pgswin(datastart - start, datastart-start+duration, np.min(dedisp_ts), 1.05*np.max(dedisp_ts))
    sp_pgplot.ppgplot.pgsch(0.8)
    sp_pgplot.ppgplot.pgslw(3)
    sp_pgplot.ppgplot.pgbox("BC", 0, 0, "BC", 0, 0)
    sp_pgplot.ppgplot.pgsci(1)
    sp_pgplot.ppgplot.pgline(times,dedisp_ts)
    errx1 = np.array([0.60 * (datastart-start+duration)])
    erry1 = np.array([0.60 * np.max(dedisp_ts)])
    erry2 = np.array([np.std(dedisp_ts)])
    errx2 = np.array([pulse_width])
    sp_pgplot.ppgplot.pgerrb(5, errx1, erry1, errx2, 1.0)
    sp_pgplot.ppgplot.pgpt(errx1, erry1, -1)
    
    # Sweeped waterfall plot Zerodm - OFF
    array = npzfile['Data_nozerodm'].astype(np.float64)
    sp_pgplot.ppgplot.pgsvp(0.20, 0.40, 0.50, 0.70)
    sp_pgplot.ppgplot.pgswin(sweeped_start, sweeped_start+sweep_duration, min_freq, max_freq)
    sp_pgplot.ppgplot.pgsch(0.8)
    sp_pgplot.ppgplot.pgslw(4)
    sp_pgplot.ppgplot.pgbox("BCST", 0, 0, "BCST", 0, 0)
    sp_pgplot.ppgplot.pgsch(3)
    sp_pgplot.plot_waterfall(array,rangex = [sweeped_start, sweeped_start+sweep_duration],rangey = [min_freq, max_freq],image = 'apjgrey')
    delays = npzfile['delays_nozerodm']
    freqs = npzfile['freqs_nozerodm']
    sp_pgplot.ppgplot.pgslw(5)
    sweepstart = sweeped_start- 0.2*sweep_duration
    sp_pgplot.ppgplot.pgsci(0)
    sp_pgplot.ppgplot.pgline(delays+sweepstart, freqs)
    sp_pgplot.ppgplot.pgsci(1)
    sp_pgplot.ppgplot.pgslw(3)
    
    # Sweeped waterfall plot Zerodm - ON
    array = npzfile['Data_zerodm'].astype(np.float64)
    sp_pgplot.ppgplot.pgsvp(0.20, 0.40, 0.1, 0.3)
    sp_pgplot.ppgplot.pgswin(sweeped_start, sweeped_start+sweep_duration, min_freq, max_freq)
    sp_pgplot.ppgplot.pgsch(0.8)
    sp_pgplot.ppgplot.pgslw(4)
    sp_pgplot.ppgplot.pgbox("BCST", 0, 0, "BCST", 0, 0)
    sp_pgplot.ppgplot.pgsch(3)
    sp_pgplot.plot_waterfall(array,rangex = [sweeped_start, sweeped_start+sweep_duration],rangey = [min_freq, max_freq],image = 'apjgrey')
    sp_pgplot.ppgplot.pgslw(5)
    sweepstart = sweeped_start- 0.2*sweep_duration
    sp_pgplot.ppgplot.pgsci(0)
    sp_pgplot.ppgplot.pgline(delays+sweepstart, freqs)
    sp_pgplot.ppgplot.pgsci(1)
    
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
    sp_pgplot.ppgplot.pgmtxt('T', -16.1, 0.01, 0.0, "Sampling time: %.3f \gms" %(tsamp*1e6))
    sp_pgplot.ppgplot.pgsvp(0.07, 0.7, 0.01, 0.05)
    sp_pgplot.ppgplot.pgmtxt('T', -2.1, 0.01, 0.0, "%s" %fn)
    
    #DM vs SNR
    dm_arr = np.float32(npzfile['dm_arr'])
    sigma_arr = np.float32 (npzfile['sigma_arr'])
    sp_pgplot.ppgplot.pgsvp(0.48, 0.73, 0.65, 0.90)
    sp_pgplot.ppgplot.pgswin(np.min(dm_arr), np.max(dm_arr), 0.95*np.min(sigma_arr), 1.05*np.max(sigma_arr))
    sp_pgplot.ppgplot.pgsch(0.8)
    sp_pgplot.ppgplot.pgslw(3)
    sp_pgplot.ppgplot.pgbox("BCNST", 0, 0, "BCNST", 0, 0)
    sp_pgplot.ppgplot.pgslw(3)
    sp_pgplot.ppgplot.pgmtxt('B', 2.5, 0.5, 0.5, "DM (pc cm\u-3\d)")
    sp_pgplot.ppgplot.pgmtxt('L', 1.8, 0.5, 0.5, "Signal-to-noise")
    sp_pgplot.ppgplot.pgpt(dm_arr, sigma_arr, 20)
    
    # DM vs Time
    spfiles = npzfile['singlepulse_files']
    threshold = 5.5
    dms, times, sigmas, files = sp_pgplot.gen_arrays(dm_arr, threshold, spfiles)
    dm_list = map(np.float32, npzfile['dm_list'])
    time_list = map(np.float32, npzfile['time_list'])
    
    sp_pgplot.dm_time_plot(dms, times, sigmas, dm_list, sigma_arr, time_list, Total_observed_time)
    sp_pgplot.ppgplot.pgiden()
    sp_pgplot.ppgplot.pgclos()
    
if __name__ == '__main__':
    parser = optparse.OptionParser(prog="waterfaller.py", \
				   usage = "%prog [OPTIONS] INFILE (.spd file)")
    parser.add_option("-x", "--xwin", action="store_true", dest="xwin",
                      default=False, help="Don't make a postscript plot, just use an X-window")
    (options, args) = parser.parse_args()
    main() 

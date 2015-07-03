#!/usr/bin/env python

"""
waterfaller.py

Make waterfall plots to show frequency sweep of a single pulse.
Reads SIGPROC filterbank format.

Patrick Lazarus - Aug. 19, 2011

"""

import sys
import optparse
import copy

import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np

import psr_utils
import rfifind

from sp_pulsar.formats import psrfits
from pypulsar.formats import filterbank
from sp_pulsar.formats import spectra

SWEEP_STYLES = ['r-', 'b-', 'g-', 'm-', 'c-']


def get_mask(rfimask, startsamp, N):
    """Return an array of boolean values to act as a mask
        for a Spectra object.

        Inputs:
            rfimask: An rfifind.rfifind object
            startsamp: Starting sample
            N: number of samples to read

        Output:
            mask: 2D numpy array of boolean values. 
                True represents an element that should be masked.
    """
    sampnums = np.arange(startsamp, startsamp+N)
    blocknums = np.floor(sampnums/rfimask.ptsperint).astype('int')
    mask = np.zeros((N, rfimask.nchan), dtype='bool')
    for blocknum in np.unique(blocknums):
        blockmask = np.zeros_like(mask[blocknums==blocknum])
        blockmask[:,rfimask.mask_zap_chans_per_int[blocknum]] = True
        mask[blocknums==blocknum] = blockmask
    return mask.T
        
def maskfile(data, start_bin, nbinsextra):
    if options.maskfile is not None:
        rfimask = rfifind.rfifind(options.maskfile) 
        mask = get_mask(rfimask, start_bin, nbinsextra)
        # Mask data
        data = data.masked(mask, maskval='median-mid80')

        datacopy = copy.deepcopy(data)
    return data
def waterfall(start_bin, dmfac, duration, nbins, zerodm, nsub, subdm, dm, integrate_dm, downsamp, scaleindep, width_bins, rawdatafile, binratio, data):
    if dm:
        nbinsextra = np.round((duration + dmfac * dm)/rawdatafile.tsamp).astype('int')
    else:
        nbinsextra = nbins

    # Zerodm filtering
    if (zerodm == True):
        data.data -=  data.data.mean(axis=0)
    
    # Subband data
    if (nsub is not None) and (subdm is not None):
        data.subband(nsub, subdm, padval='mean')

    # Dedisperse
    if dm:
        data.dedisperse(dm, padval='mean')

    # Downsample
    data.downsample(downsamp)

    # scale data
    data = data.scaled(scaleindep)
    
    # Smooth
    if width_bins > 1:
        data.smooth(width_bins, padval='mean')

    return data, nbinsextra
def main():
    fn = args[0]
    if fn.endswith(".fil"):
        # Filterbank file
        filetype = "filterbank"
        rawdatafile = filterbank.filterbank(fn)
    elif fn.endswith(".fits"):
        # PSRFITS file
        filetype = "psrfits"
        rawdatafile = psrfits.PsrfitsFile(fn)
    else:
        raise ValueError("Cannot recognize data file type from "
                         "extension. (Only '.fits' and '.fil' "
                         "are supported.)")

    # Read data
    start_bin = np.round(options.start/rawdatafile.tsamp).astype('int')
	#dmfac = 4.15e3 * np.abs(1./rawdatafile.fch1**2 - 1./(rawdatafile.frequencies[-1])**2)
    dmfac = 4.15e3 * np.abs(1./rawdatafile.frequencies[0]**2 - 1./rawdatafile.frequencies[-1]**2)
    if options.nbins is None:
        nbins = np.round(options.duration/rawdatafile.tsamp).astype('int')
    else:
        nbins = options.nbins
    binratio = 50    
    if options.dm:
	nbinsextra = np.round((options.duration + dmfac * options.dm)/rawdatafile.tsamp).astype('int')
    else:
        nbinsextra = nbins    
    data = rawdatafile.get_spectra(start_bin, nbinsextra)
    data = maskfile(data, start_bin, nbinsextra)
    data, bins = waterfall(start_bin, dmfac, options.duration, nbins, options.zerodm, options.nsub, options.subdm, options.dm, options.integrate_dm, options.downsamp, options.scaleindep, options.width_bins, rawdatafile, binratio, data)
    # Ploting it up
    fig = plt.figure()
    fig.canvas.set_window_title("Frequency vs. Time")
    ax = plt.axes((0.15, 0.15, 0.8, 0.60))
    ragfac = float(nbins)/bins
    dmrange, trange = data.data.shape
    nbinlim = np.int(trange * ragfac)
    print data.dt, rawdatafile.tsamp 
   #np.save('data',data.data[..., :nbinlim])
    plt.imshow(data.data[..., :nbinlim], aspect='auto', \
                cmap=matplotlib.cm.cmap_d[options.cmap], \
                interpolation='nearest', origin='upper', \
                extent=(data.starttime, data.starttime+ nbinlim*data.dt, \
                        data.freqs.min(), data.freqs.max()))
    if options.show_cb:
        cb = plt.colorbar()
        cb.set_label("Scaled signal intensity (arbitrary units)")

    #plt.axis('tight')
    # Sweeping it up
    for ii, sweep_dm in enumerate(options.sweep_dms):
        ddm = sweep_dm-data.dm
        delays = psr_utils.delay_from_DM(ddm, data.freqs)
        delays -= delays.min()
        
        if options.sweep_posns is None:
            sweep_posn = 0.0
        elif len(options.sweep_posns) == 1:
            sweep_posn = options.sweep_posns[0]
        else:
            sweep_posn = options.sweep_posns[ii]
        sweepstart = data.dt*data.numspectra*sweep_posn+data.starttime
        sty = SWEEP_STYLES[ii%len(SWEEP_STYLES)]
        plt.plot(delays+sweepstart, data.freqs, sty, lw=4, alpha=0.5)

    # Dressing it up
    plt.xlabel("Time")
    plt.ylabel("Observing frequency (MHz)")
    plt.suptitle("Frequency vs. Time")
    # Plot Time series
    if options.integrate_dm is not None:
        Data = np.array(data.data[..., :nbinlim])
        Dedisp_ts = Data.sum(axis=0)
        times = (np.arange(data.numspectra)*data.dt + options.start)[..., :nbinlim]
        ax = plt.axes((0.15, 0.75, 0.8, 0.2))
        plt.plot(times, Dedisp_ts)
	plt.setp(ax.get_xticklabels(), visible = False)
	plt.setp(ax.get_yticklabels(), visible = False)
    fig.canvas.mpl_connect('key_press_event', \
            lambda ev: (ev.key in ('q','Q') and plt.close(fig)))
    plt.show()


if __name__=='__main__':
    parser = optparse.OptionParser(prog="waterfaller.py", \
                        version="v0.9 Patrick Lazarus (Aug. 19, 2011)", \
                        usage="%prog [OPTIONS] INFILE", \
                        description="Create a waterfall plot to show the " \
                                    "frequency sweep of a single pulse " \
                                    "in psrFits data.")
    parser.add_option('--subdm', dest='subdm', type='float', \
                        help="DM to use when subbanding. (Default: " \
                                "same as --dm)", default=None)
    parser.add_option('--zerodm', dest='zerodm', action='store_true', \
                        help="If this flag is set - Turn Zerodm filter - ON  (Default: " \
                                "OFF)", default=False)
    parser.add_option('-s', '--nsub', dest='nsub', type='int', \
                        help="Number of subbands to use. Must be a factor " \
                                "of number of channels. (Default: " \
                                "number of channels)", default=None)
    parser.add_option('-d', '--dm', dest='dm', type='float', \
                        help="DM to use when dedispersing data for plot. " \
                                "(Default: 0 pc/cm^3)", default=0.0)
    parser.add_option('--integratedm', dest='integrate_dm', type='float', \
                        help="DM to use when plotting the dedispersed time series. " \
                                "(Default: Do not show the time series", default=None)
    parser.add_option('-T', '--start-time', dest='start', type='float', \
                        help="Time into observation (in seconds) at which " \
                                "to start plot.")
    parser.add_option('-t', '--duration', dest='duration', type='float', \
                        help="Duration (in seconds) of plot.")
    parser.add_option('-n', '--nbins', dest='nbins', type='int', \
                        help="Number of time bins to plot. This option takes " \
                                "precedence over -t/--duration if both are " \
                                "provided.")
    parser.add_option('--width-bins', dest='width_bins', type='int', \
                        help="Smooth each channel/subband with a boxcar " \
                                "this many bins wide. (Default: Don't smooth)", \
                        default=1)
    parser.add_option('--sweep-dm', dest='sweep_dms', type='float', \
                        action='append', \
                        help="Show the frequency sweep using this DM. " \
                                "(Default: Don't show sweep)", default=[])
    parser.add_option('--sweep-posn', dest='sweep_posns', type='float', \
                        action='append', \
                        help="Show the frequency sweep at this position. " \
                                "The position refers to the high-frequency " \
                                "edge of the plot. Also, the position should " \
                                "be a number between 0 and 1, where 0 is the " \
                                "left edge of the plot. "
                                "(Default: 0)", default=None)
    parser.add_option('--downsamp', dest='downsamp', type='int', \
                        help="Factor to downsample data by. (Default: 1).", \
                        default=1)
    parser.add_option('--mask', dest='maskfile', type='string', \
                        help="Mask file produced by rfifind. (Default: No Mask).", \
                        default=None)
    parser.add_option('--scaleindep', dest='scaleindep', action='store_true', \
                        help="If this flag is set scale each channel " \
                                "independently. (Default: Scale using " \
                                "global maximum.)", \
                        default=False)
    parser.add_option('--show-colour-bar', dest='show_cb', action='store_true', \
                        help="If this flag is set show a colour bar. " \
                                "(Default: No colour bar.)", \
                        default=False)
    parser.add_option('--colour-map', dest='cmap', \
                        help="The name of a valid matplotlib colour map." \
                                "(Default: gist_yarg.)", \
                        default='gist_yarg')
    options, args = parser.parse_args()
    
    if not hasattr(options, 'start'):
        raise ValueError("Start time (-T/--start-time) " \
                            "must be given on command line!")
    if (not hasattr(options, 'duration')) and (not hasattr(options, 'nbins')):
        raise ValueError("One of duration (-t/--duration) " \
                            "and num bins (-n/--nbins)" \
                            "must be given on command line!")
    if options.subdm is None:
        options.subdm = options.dm
    main()

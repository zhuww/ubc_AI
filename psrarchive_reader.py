import sys,os
import scipy.stats as stats
import numpy as np
import ubc_AI.samples

def rotate(data, deltaphase): 
    size = data.shape[-1]
    deltabin = np.round(size * deltaphase)
    return np.roll(data, int(deltabin), axis=-1)

def calDMcurve(data2d, ddms, freqs, period):
    chisqs = []
    for i,ddm in enumerate(ddms):
        deltaphases = ddm * 4.15e3 * 1. / freqs**2 / period
        data = np.array([rotate(data2d[j,:], dp) for j,dp in enumerate(deltaphases)])
        chisqs.append(stats.chisquare(data.sum(0))[0])
    return np.array(chisqs)

def greyscale(img):
    global_max = np.maximum.reduce(np.maximum.reduce(img))
    min_parts = np.minimum.reduce(img, 1)
    img = (img-min_parts[:,np.newaxis])/global_max
    return img

class ar2data(object):
    initialize = False
    def __init__(self, filename, align=True, centre=True):
        try:
            import psrchive
        except:
            print '''cannot load the psrchive python module
            make sure you have psrchive installed with the 
            configure --enable-shared option'''
            raise Error
        self.filename = filename
        self.archive = psrchive.Archive_load(filename)
        #archive.bscrunch_to_nbin(64)
        self.archive.dedisperse()
        self.archive.remove_baseline()
        data = self.archive.get_data()
        self.data = data[:,0,:,:]
        archive0 = self.archive[0]
        self.dm = self.archive.get_dispersion_measure()
        self.freq = self.archive.get_centre_frequency()
        self.bandwidth = np.abs(self.archive.get_bandwidth())
        self.freq_lo = self.freq - self.bandwidth/2.
        self.freq_hi = self.freq + self.bandwidth/2.
        self.freqbins = self.data.shape[1]
        self.binwidth = self.bandwidth/self.freqbins
        self.freqs = np.mgrid[self.freq_lo+self.binwidth/2.:self.freq_hi-self.binwidth/2.:self.freqbins*1j]
        self.period = archive0.get_folding_period()
        dmfac = 4.15e3 * np.abs(1./self.freqs.min()**2 - 1./self.freqs.max()**2) 
        ddm = 2. * self.period / dmfac
        lowdm = max(0, self.dm-ddm)
        hidm = self.dm+ddm
        self.dms = np.linspace(lowdm, hidm, 50)
        self.profile = self.data.sum(0).sum(0)
        mx = self.profile.argmax()
        if centre:
            nbin = self.profile.size
            noff = nbin/2 - mx
            self.data = np.roll(self.data, noff, axis=-1)
        if align:
            self.align = mx
        else:
            self.align = 0
        self.initialize = True
        
    def getdata(self, phasebins=0, freqbins=0, timebins=0, DMbins=0, intervals=0, subbands=0, bandpass=0, ratings=None):
        """
        input: feature=feature_size
        possible features:
            phasebins: summmed profile, data cube (self.profs) summed(projected) to the phase axis.
            freqbins: summed frequency profile, data cube projected to the frequency axis
            timebins: summed time profile, data cube projected to the time axis.
            DMbins: DM curves.
            intervals: the time vs phase image
            subbands: the subband vs phase image
            ratings: List of possible rating stored in the pfd file, possible values including: period, redchi2, offredchi2, avgvoverc
        usage examples:

        """
        if not 'extracted_feature' in self.__dict__:
            self.extracted_feature = {}
        data = self.data

        normalize = ubc_AI.samples.normalize
        downsample = ubc_AI.samples.downsample

        def getsumprofs(M):
            feature = '%s:%s' % ('phasebins', M)
            if M == 0:
                return np.array([])
            prof = normalize(data).sum(0).sum(0)
            result = normalize(downsample(prof,M,align=self.align).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getfreqprofs(M):
            feature = '%s:%s' % ('freqbins', M)
            if M == 0:
                return np.array([])
            prof = normalize(data).sum(0).sum(1)
            result = normalize(downsample(data,M,align=self.align).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def gettimeprofs(M):
            feature = '%s:%s' % ('timebins', M)
            if M == 0:
                return np.array([])
            prof = normalize(data).sum(1).sum(1)
            result = normalize(downsample(data,M,align=self.align).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getbandpass(M):
            feature = '%s:%s' % ('bandpass', M)
            if M == 0:
                return np.array([])
            prof = normalize(data).sum(0).sum(1)
            result = normalize(downsample(data,M,align=self.align).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getDMcurve(M):
            feature = '%s:%s' % ('DMbins', M)
            if M == 0:
                return np.array([])
            chisqs = calDMcurve(self.data.sum(0), self.dms - self.dm, self.freqs, self.period)
            result = normalize(downsample(chisqs,M).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getintervals(M):
            feature = '%s:%s' % ('intervals', M)
            if M == 0:
                return np.array([])
            img = greyscale(data.sum(1))
            result = downsample(normalize(img),M,align=self.align).ravel()
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getsubbands(M):
            feature = '%s:%s' % ('subbands', M)
            if M == 0:
                return np.array([])
            img = greyscale(data.sum(0))
            result = downsample(normalize(img),M,align=self.align).ravel()
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getratings(L):
            feature = '%s:%s' % ('ratings', L)
            if L == None:
                return np.array([])
            if not feature in self.extracted_feature:
                result = []
                for rating in L:
                    if rating == 'period':
                        result.append(self.period)
                    elif rating == 'dm':
                        result.append(self.dm)
                    else:
                        result.append(self.__dict__[rating])
                self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        data = np.hstack((getsumprofs(phasebins), getfreqprofs(freqbins), gettimeprofs(timebins), getbandpass(bandpass), getDMcurve(DMbins), getintervals(intervals), getsubbands(subbands), getratings(ratings)))
        return data



if __name__ == '__main__':
    from pylab import *
    ar2file = ar2data('test.ar2', align=True)

    data = ar2file.getdata(intervals=64)
    imshow(data.reshape((64,64)), aspect='auto')
    show()

    #data = ar2file.getdata(subbands=64)
    #imshow(data.reshape((64,64)), aspect='auto')
    #show()

    data = ar2file.getdata(phasebins=32)
    plot(data, '-')
    show()

    #data = ar2file.getdata(DMbins=16) 
    #plot(data, '.')
    #show()

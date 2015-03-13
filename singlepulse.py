import numpy as np
import os,sys
import scipy.stats as stats
import ubc_AI.samples
DM_range_factor = 0.2


def rotate(data, deltaphase): 
    size = data.shape[-1]
    deltabin = np.round(size * deltaphase)
    return np.roll(data, int(deltabin), axis=-1)

def calDMcurve(data2d, freqs, dm, period):
    dmfac = 4.15e3 * np.abs(1./freqs.min()**2 - 1./freqs.max()**2) 
    ddm = DM_range_factor * period / dmfac
    lowdm = max(0, dm-ddm)
    hidm = dm+ddm
    dms = np.linspace(lowdm, hidm, 100)
    ddms = dms - dm
    chisqs = []
    data2d.shape[0]
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

class waterfall(object): 
    initialize = False
    def __init__(self, data, dm, duration, freq_lo, freq_hi, align=True, centre=True):
        self.data = data
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.dm = dm
        self.duration = duration
        self.profile = self.data.sum(0)
        mx = self.profile.argmax()
        if centre:
            nbin = self.profile.size
            noff = nbin/2 - mx
            self.data = np.roll(self.data, noff, axis=-1)
        if align:
            self.align = mx
        else:
            self.align = 0
        self.extracted_feature = {}
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
            prof = normalize(data).sum(0)
            result = normalize(downsample(prof,M,align=self.align).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getfreqprofs(M):
            feature = '%s:%s' % ('freqbins', M)
            if M == 0:
                return np.array([])
            prof = normalize(data).sum(1)
            result = normalize(downsample(data,M,align=self.align).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def gettimeprofs(M):
            feature = '%s:%s' % ('timebins', M)
            if M == 0:
                return np.array([])
            prof = normalize(data).sum(0)
            result = normalize(downsample(data,M,align=self.align).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getbandpass(M):
            feature = '%s:%s' % ('bandpass', M)
            if M == 0:
                return np.array([])
            prof = normalize(data).sum(1)
            result = normalize(downsample(data,M,align=self.align).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getDMcurve(M):
            feature = '%s:%s' % ('DMbins', M)
            if M == 0:
                return np.array([])
            fbins = self.data.shape[0]
            newfreqs = np.mgrid[self.freq_lo:self.freq_hi:fbins*1j]
            chisqs = calDMcurve(data, newfreqs, self.dm, self.duration)
            #chisqs = calDMcurve(self.data.sum(0), self.dms - self.dm, self.freqs, self.period)
            result = normalize(downsample(chisqs,M).ravel())
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getintervals(M):
            feature = '%s:%s' % ('intervals', M)
            if M == 0:
                return np.array([])
            img = greyscale(data)
            result = downsample(normalize(img),M,align=self.align).ravel()
            self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        def getsubbands(M):
            feature = '%s:%s' % ('subbands', M)
            if M == 0:
                return np.array([])
            img = greyscale(data)
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
                    if rating == 'duration':
                        result.append(self.duration)
                    elif rating == 'period':
                        result.append(self.duration)
                    elif rating == 'dm':
                        result.append(self.dm)
                    else:
                        result.append(self.__dict__[rating])
                self.extracted_feature[feature] = np.array(result)
            return self.extracted_feature[feature]
        data = np.hstack((getsumprofs(phasebins), getfreqprofs(freqbins), gettimeprofs(timebins), getbandpass(bandpass), getDMcurve(DMbins), getintervals(intervals), getsubbands(subbands), getratings(ratings)))
        return data

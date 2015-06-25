import cPickle, glob, ubc_AI
import os, sys
from ubc_AI.data import pfdreader
import numpy as np
from ubc_AI.ProgressBar import progressBar as PB
AI_PATH = '/'.join(ubc_AI.__file__.split('/')[:-1])
classifier = cPickle.load(open(AI_PATH+'/trained_AI/clfl2_PALFA.pkl','rb'))
pfdfile = set(glob.glob('*.pfd') + glob.glob('*.ar') + glob.glob('*.ar2'))
clfedfiles = set(np.genfromtxt('clfresult.txt', dtype=[('files', '|S200'), ('AI', 'f8')])['files'])
#clfedfiles = set([])
targetfiles = list(pfdfile - clfedfiles)

fbatchs = np.array_split(np.array(targetfiles), 1000)
Nbatchs = len(fbatchs)

pb = PB(maxValue=Nbatchs)

fout = open('clfresult.txt', 'a')
#ferr = open('badlist.txt', 'a')
for i in range(Nbatchs):
    files = fbatchs[i]
    try:
        AI_scores = classifier.report_score([pfdreader(f) for f in files])
        text = ''.join(['%s %s\n' % (files[j], AI_scores[j]) for j in range(len(files))]) 
        fout.write(text)
    except:
        for f in files:
            try:
                s = classifier.report_score([pfdreader(f)])[0]
                text = '%s %s\n' % (f, s)
                fout.write(text)
            except:
                #text = '%s\n' % f
                #ferr.write(text)
                continue
    pb(i+1)
fout.close()
#ferr.close()

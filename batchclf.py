import cPickle, glob, ubc_AI
from ubc_AI.data import pfdreader
import numpy as np
AI_PATH = '/'.join(ubc_AI.__file__.split('/')[:-1])
classifier = cPickle.load(open(AI_PATH+'/trained_AI/clfl2_PALFA.pkl','rb'))
#classifier = cPickle.load(open(AI_PATH+'/trained_AI/clfl2_BD.pkl','rb'))
pfdfile = glob.glob('*.pfd') + glob.glob('*.ar') + glob.glob('*.ar2') + glob.glob('*.spd')

#AI_scores = classifier.report_score([pfdreader(f) for f in pfdfile])

#batch_size = 100
#N_pfds = len(pfdfile)
#n_batchs = int(N_pfds / batch_size)
#pfd_batchs = np.array_split(pfdfile, n_batchs)

badlist = []
AI_scores = []

for pfd in pfdfile:
    try:
        score = classifier.report_score([pfdreader(pfd)])[0]
    except:
        print 'corrupted file:', pfd
        badlist.append(pfd)
        score = -1
        classifier = cPickle.load(open(AI_PATH+'/trained_AI/clfl2_PALFA.pkl','rb'))
    AI_scores.append(score)


#for pfds in pfd_batchs:
    #try:
        #batch_scores = classifier.report_score([pfdreader(f) for f in pfds])
    #except:
        #classifier = cPickle.load(open(AI_PATH+'/trained_AI/clfl2_PALFA.pkl','rb'))
        #batch_scores = []
        #for pfd in pfds:
            #try:
                #batch_scores.append(classifier.report_score([pfdreader(pfd)])[0])
            #except:
                #print 'corrupted file:', pfd
                #badlist.append(pfd)
                #batch_scores.append(-1)
                #classifier = cPickle.load(open(AI_PATH+'/trained_AI/clfl2_PALFA.pkl','rb'))

    #AI_scores.extend(batch_scores)


text = '\n'.join(['%s %s' % (pfdfile[i], AI_scores[i]) for i in range(len(pfdfile))])
fout = open('clfresult.txt', 'w')
fout.write(text)
fout.close()

fbad = open('badpfdlist.txt', 'w')
fbad.write('\n'.join(badlist))
fbad.close()

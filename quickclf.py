import cPickle, glob, ubc_AI
from ubc_AI.data import pfdreader
AI_PATH = '/'.join(ubc_AI.__file__.split('/')[:-1])
classifier = cPickle.load(open(AI_PATH+'/trained_AI/clfl2_PALFA.pkl','rb'))
pfdfile = glob.glob('*.pfd') + glob.glob('*.ar') + glob.glob('*.ar2') + glob.glob('*.spd')
AI_scores = classifier.report_score([pfdreader(f) for f in pfdfile])

text = '\n'.join(['%s %s' % (pfdfile[i], AI_scores[i]) for i in range(len(pfdfile))])
fout = open('clfresult.txt', 'w')
fout.write(text)
fout.close()

"""
Aaron Berndsen: a module which reads the ATNF database and 
GBNCC discoveries page and provides a list of known pulsars.

Pulsars are given as ephem.FixedBody objects, with all the 
ATNF pulsar properties in self.props[key].

"""

import numpy as np
import urllib

from BeautifulSoup import BeautifulSoup

def get_allpulsars():
    """
    return dictionary of all pulsars found in 
    ATNF, PALFA and GBNCC databases
    """
    allpulsars = {}
    atnf = ATNF_pulsarlist()
    gbncc = GBNCC_pulsarlist()
    palfa = PALFA_pulsarlist()
    for k, v in atnf.iteritems():
        allpulsars[k] = v
    for k, v in gbncc.iteritems():
        if k in allpulsars:
            print "GBNCC %s may be in ATNF database already" % k
        allpulsars[k] = v
    for k, v in palfa.iteritems():
        if k in allpulsars:
            if ('%4.3f' % allpulsars[k].P0) == ('%4.3f' % v.P0):
                print "PALFA psr %s is already in ATNF database" % k
                continue
        allpulsars[k] = v
    return allpulsars

def ATNF_pulsarlist():
        """
        Contact the ATNF pulsar catalogue, returning an array of data.
        Each row corresponding to one pulsar, with columns in the format:

        return:
        dictionary of pulsars, keyed by PSRJ
        """
        #URL to get | NAME | PSRJ | RAJ | DECJ | P0 | DM |
        url = 'http://www.atnf.csiro.au/research/pulsar/psrcat/proc_form.php?Name=Name&JName=JName&RaJ=RaJ&DecJ=DecJ&P0=P0&DM=DM&startUserDefined=true&c1_val=&c2_val=&c3_val=&c4_val=&sort_attr=jname&sort_order=asc&condition=&pulsar_names=&ephemeris=short&coords_unit=raj%2Fdecj&radius=&coords_1=&coords_2=&style=Short+without+errors&no_value=*&fsize=3&x_axis=&x_scale=linear&y_axis=&y_scale=linear&state=query&table_bottom.x=27&table_bottom.y=10'
        sock = urllib.urlopen(url)
        data = sock.read()
        sock.close()
        data = data.split('<pre>')[1]
        data = data.split('</pre>')[0]
        data = data.splitlines()[5:-1]
        data = [data[i].split()[1:] for i in range(len(data)) if len(data[i]) > 1]
        
        pulsars = {}
        for line in data:
            p = pulsar(*line)
            pulsars[p.psrj] = p

        return pulsars

def GBNCC_pulsarlist():
    """
    gather the pulsars listed on the GBNCC discover page

    return:
    dictionary of pulsars keyed by PSRJ
    """
    url = 'http://arcc.phys.utb.edu/gbncc/'
    sock = urllib.urlopen(url)
    data = sock.read()
    soup = BeautifulSoup(data)
    sock.close()
    rows = soup.findAll('tr')[1:]
    pulsars = {}
    for row in rows:
        cols = row.findAll('td')
        name = cols[0].text
        p0 = cols[1].text
        dm = cols[2].text
        coords = name.strip('J')
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            decj = str('+%s:00' % coords.split('+')[1])
        else:
            raj = coords.split('')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            decj = str('-%s:00' % coords.split('-')[1])

        pulsars[name] = pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=True)

    return pulsars

def PALFA_pulsarlist():
    """
    gather the pulsars listed on the palfa discover page

    return:
    dictionary of pulsars keyed by PSRJ
    """
    url = 'http://www.naic.edu/~palfa/newpulsars/'
    sock = urllib.urlopen(url)
    data = sock.read()
    soup = BeautifulSoup(data)
    sock.close()
    table = soup.findAll('table')[0] #pulsars are in first table
    rows = table.findAll('tr')[1:]
    pinfo = PALFA_jodrell_extrainfo()
    pulsars = {}
    for row in rows:
        cols = row.findAll('td')
        name = cols[1].text
        if name == 'J2010+31': continue #skip single-pulse 
        try:
            p0 = np.float(cols[2].text.strip('~'))/1000. #[ms]-->[s]
        except:
            p0 = np.nan
        if pinfo.has_key(name):
            if int(pinfo[name][0]*1000) == int(p0*1000):
                dm = pinfo[name][1]
            else:
                dm = np.nan
        else:
            dm = np.nan 
        coords = name.strip('J')
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            decj = str('+%s:00' % coords.split('+')[1])
        else:
            raj = coords.split('')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            decj = str('-%s:00' % coords.split('-')[1])

        pulsars[name] = pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=True)

    return pulsars

def PALFA_jodrell_extrainfo():
    """
    return a dictionary keyed by the pulsar name, with values
    (p0, DM). 
    This is because http://www.naic.edu/~palfa/newpulsars/
    doesn't list the DM, but 
    http://www.jodrellbank.manchester.ac.uk/research/pulsar/PALFA/
    does.

    """
    url = 'http://www.jodrellbank.manchester.ac.uk/research/pulsar/PALFA/'
    sock = urllib.urlopen(url)
    data = sock.read()
    soup = BeautifulSoup(data)
    sock.close()
    table = soup.findAll('table')[0] #pulsars are in first table
    rows = table.findAll('tr')[1:]
    pulsars = {}
    for row in rows:
        cols = row.findAll('td')
        name = 'J%s' % cols[0].text.strip('_jb.tim')
        dm = np.float(cols[3].text)
        p0 = np.float(cols[2].text)/1000. #[ms] --> [s]
        coords = name.strip('J')
        pulsars[name] = (p0, dm)
    return pulsars


#               
#    ______  __ __|  |   ___________ _______ 
#    \____ \|  |  \  |  /  ___/\__  \\_  __ \
#    |  |_> >  |  /  |__\___ \  / __ \|  | \/
#    |   __/|____/|____/____  >(____  /__|   
#    |__|                   \/      \/       
#
class pulsar():
    """
    A simple class describing a pulsar. 
    Only tracks: name, ra, dec, P0, DM
   
   	ra/dec should be in HH:MM:SS or DD:MM:SS format

    """
    def __init__(self, name, psrj, ra, dec, P0, DM, gbncc=False):
        self.name = name
        self.psrj = psrj
        self.ra = ra 
        self.dec = dec 
        #keep track if this was listed on GBNCC page
        #since it only gives DEC to nearest degree (used in matching)
        self.gbncc = gbncc
        try:
            self.P0 = float(P0)
        except ValueError:
            self.P0 = np.nan
        try:
            self.DM = float(DM)
        except:
            self.DM = np.nan

def hhmm2deg(hhmm):
    """
    given string 'hhmmss' or 'hh:mm:ss' (of varying length), convert to 
    degrees

    """
    if ':' in hhmm:
        s = hhmm.split(':')
        ncomp = len(s)
        if ncomp == 1:
            deg = float(s[0])*360./24.
        elif ncomp == 2:
            deg = float(s[0])*360./24. + float(s[1])/60.
        elif ncomp >= 3:
            deg = float(s[0])*360./24. + float(s[1])/60. + float(s[2])/3600.
    else:
        if len(hhmm) == 2:
            deg = float(hhmm)*360./24.
        elif len(hhmm) == 4:
            deg = float(hhmm[0:2])*360./24. + float(hhmm[2:4])/60.
        elif len(hhmm) >= 6:
            deg = float(hhmm[0:2])*360./24. + float(hhmm[2:4])/60.\
                + float(hhmm[4:6])/3600.
    return deg

def ddmm2deg(ddmm):
    """
    given string 'hhmmss' or 'hh:mm:ss', convert to 
    degrees

    """
    if '-' in ddmm:
        sgn = -1
    else:
        sgn = 1

    if ':' in ddmm:
        s = ddmm.split(':')
        ncomp = len(s)
        if ncomp == 1:
            deg = abs(float(s[0]))
        elif ncomp == 2:
            deg = abs(float(s[0])) + float(s[1])/60.
        elif ncomp >= 3:
            deg = abs(float(s[0])) + float(s[1])/60. + float(s[2])/3600.
    else:
        if len(ddmm) == 2:
            deg = abs(float(ddmm))
        elif len(ddmm) == 4:
            deg = abs(float(ddmm[0:2])) + float(ddmm[2:4])/60.
        elif len(ddmm) >= 6:
            deg = abs(float(ddmm[0:2])) + float(ddmm[2:4])/60.\
                + float(ddmm[4:6])/3600.
    return deg*sgn
    
def matches(allpulsars, pulsar, sep=.5):
    """
    given a dictionary of all pulsars, return
    the objects within 'sep' degrees of the pulsar.

    args:
    allpulsars : dictionary of allpulsars
    pulsar : object of interest
    sep : degrees of separation [default 30'=.5deg]


    Notes:
    because GBNCC only gives Dec to nearest degree, we consider
    things close if they are separated by less than a degree in dec,
    and 'sep' in RA
    """
    matches = {}
    pra = hhmm2deg(pulsar.ra)
    pdec = ddmm2deg(pulsar.dec)
    for k, v in allpulsars.iteritems():
        ra = hhmm2deg(v.ra)
        dec = ddmm2deg(v.dec)
        if pulsar.gbncc:
            if abs(ra - pra) <= sep and \
                    abs(dec - pdec) <= 1:
                matches[k] = v
        elif abs(ra - pra) <= sep and \
                abs(dec - pdec) <= sep:
            matches[k] = v
    return sorted(matches.values(), key=lambda x: x.name, reverse=True) #gives a sorted list
    
        
    

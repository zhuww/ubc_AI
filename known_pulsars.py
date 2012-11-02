"""
Aaron Berndsen: a module which reads the ATNF database and 
GBNCC discoveries page and provides a list of known pulsars.

Pulsars are given as ephem.FixedBody objects, with all the 
ATNF pulsar properties in self.props[key].

"""

import numpy as np
import re
import urllib

from BeautifulSoup import BeautifulSoup

def get_allpulsars():
    """
    return dictionary of all pulsars found in 
    ATNF, PALFA and GBNCC databases
    """
    allpulsars = {}
    atnf = ATNF_pulsarlist()
    two_join = {'gbncc': GBNCC_pulsarlist(),
                'palfa': PALFA_pulsarlist(),
                'drift' : driftscan_pulsarlist(),
                'ao327' : ao327_pulsarlist(),
                'deepmb' : deepmb_pulsarlist(),
                'gbt350' : GBT350NGP_pulsarlist(),
                'fermi' : FERMI_pulsarlist(),
                }

    for k, v in atnf.iteritems():
        allpulsars[k] = v
    for survey, lst in two_join.iteritems():
        for k, v in lst.iteritems():
            if k in allpulsars:
                #add it in if P0 differ by 10ms
                if int(100*allpulsars[k].P0) == int(100*v.P0):
                    print "pulsar %s from %s already in our database" % (k, survey)
                    continue
                else:
                    k += 'a'
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
            p = pulsar(*line,catalog='ATNF')
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
        p0 = float(cols[1].text)/1000. #ms --> [s]
        dm = cols[2].text
        coords = name.strip('J')   
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
        else:
            raj = coords.split('')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))

        pulsars[name] = pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=True,
                               catalog=url)

    return pulsars

def PALFA_pulsarlist():
    """
    gather the pulsars listed on the palfa discover page
    http://www.naic.edu/~palfa/newpulsars/

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
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
        else:
            raj = coords.split('')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))

        pulsars[name] = pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=True,
                               catalog=url)

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


def driftscan_pulsarlist():
    url = 'http://www.as.wvu.edu/~pulsar/GBTdrift350/'
    sock = urllib.urlopen(url)
    data = sock.read()
    soup = BeautifulSoup(data)
    sock.close()
    table = soup.findAll('table')[0] #pulsars are in first table
    rows = table.findAll('tr')
    pulsars = {}
    for row in rows:
        cols = row.findAll('td')
        if len(cols) == 0: continue

        name = str(cols[0].text)
        p0 = np.float(cols[1].text)/1000. #[ms] --> [s]
        dm = np.float(cols[2].text)
        coords = name.strip('J')
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
        else:
            raj = coords.split('-')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))

        pulsars[name] =  pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=True,
                                catalog=url)
    return pulsars

def ao327_pulsarlist():
    url = 'http://www.naic.edu/~deneva/drift-search/'
    sock = urllib.urlopen(url)
    data = sock.read()
    soup = BeautifulSoup(data)
    sock.close()
    table = soup.findAll('table')[0] #pulsars are in first table
    rows = table.findAll('tr')[1:]
    pulsars = {}
    for row in rows:
        cols = row.findAll('td')
        name = str(cols[1].text)
        try:
            p0 = np.float(cols[2].text)/1000. #[ms] --> [s]
        except ValueError:
            p0 = np.nan
        dm = np.float(cols[3].text.strip('~'))
        coords = name.strip('J')
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
        else:
            raj = coords.split('-')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))

        pulsars[name] =  pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=True,
                                catalog=url)
    return pulsars

def deepmb_pulsarlist():
    url = 'http://astro.phys.wvu.edu/dmb/'
    sock = urllib.urlopen(url)
    data = sock.read()
    soup = BeautifulSoup(data)
    sock.close()
    table = soup.findAll('table')[0] #pulsars are in first table
    rows = table.findAll('tr')[1:]
    pulsars = {}
    for row in rows:
        cols = row.findAll('td')
        name = str(cols[0].text)
        p0 = np.float(cols[1].text)/1000. #[ms] --> [s]
        dm = np.float(cols[2].text.strip('~'))
        coords = name.strip('J')
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
        else:
            raj = coords.split('-r')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
        pulsars[name] =  pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=True,
                                catalog=url)
    return pulsars


def FERMI_pulsarlist():
    """
    pulsars found in :
    http://arxiv.org/pdf/1205.3089v1.pdf

    """
    url = 'http://arxiv.org/pdf/1205.3089v1.pdf'
    lst = ["J0023+0923 GBT-350 1FGLJ0023.5+0930 3.05 14.3 0.7 0.14 0.017 BW, ",
           "J0101-6422 Parkes 1FGLJ0101.06423 2.57 12.0 0.6 1.78 0.16 ",
           "J0102+4839 GBT-350 1FGLJ0103.1+4840 2.96 53.5 2.3 1.67 0.18 ",
           "J0307+7443 GBT-350 1FGLJ0308.6+7442 3.16 6.4 0.6 36.98 0.24 ",
           "J0340+4130 GBT-350 1FGLJ0340.4+4130 3.30 49.6 1.8 Isolated ",
           "J0533+67 GBT-820 2FGLJ0533.9+6759 4.39 57.4 2.4 Isolated",
           "J0605+37 GBT-820 2FGLJ0605.3+3758 2.73 21.0 0.7 55.6 0.18",
           "J0614-3329 GBT-820 1FGLJ0614.13328 3.15 37.0 1.9 53.6 0.28 ",
           "J0621+25 GBT-820 2FGLJ0621.2+2508 2.72 83.6 2.3 TBD",
           "J1103-5403 Parkes 1FGLJ1103.95355 3.39 103.9 2.5 Isolated NA [18]",
           "J1124-3653 GBT-350 1FGLJ1124.43654 2.41 44.9 1.7 0.23 0.027 BW, ",
           "J1124-36 GMRT-325 1FGLJ1124.43654 5.55 45.1 1.7 TBD NA",
           "J1142+0119 GBT-820 1FGLJ1142.7+0127 5.07 19.2 0.9 1.58 0.15",
           "J1207-5050 GMRT-610 1FGLJ1207.05055 4.84 50.6 1.5 Isolated?",
           "J1231-1411 GBT-820 1FGLJ1231.11410 3.68 8.1 0.4 1.86 0.19 ",
           "J1301+0833 GBT-820 1FGLJ1301.8+0837 1.84 13.2 0.7 0.27 0.024 BW",
           "J1302-32 GBT-350 1FGLJ1302.33255 3.77 26.2 1.0 0.78 0.15 [16]",
           "J1312+0051 GBT-820 1FGLJ1312.6+0048 4.23 15.3 0.8 38.5 0.18 ",
           "J1514-4946 Parkes 1FGLJ1514.14945 3.59 30.9 0.9 1.92 0.17 ",
           "J1536-4948 GMRT-325 1FGLJ1536.54949 3.08 38.0 1.8 TBD",
           "J1544+4937 GMRT-610 18M3037 2.16 23.2 1.2 0.117 0.018 BW [7]",
           "J1551-06x GBT-350 1FGLJ1549.70659 7.09 21.6 1.0 5.21 0.20 [16]",
           "J1628-3205 GBT-820 1FGLJ1627.83204 3.21 42.1 1.2 0.21 0.16 RB",
           "J1630+37 GBT-820 2FGLJ1630.3+3732 3.32 14.1 0.9 12.5 0.16",
           "J1646-2142 GMRT-325 1FGLJ1645.02155c 5.85 29.8 1.1 23 TBD",
           "J1658-5324 Parkes 1FGLJ1658.85317 2.43 30.8 0.9 Isolated ",
           "J1745+1017 Elsberg 1FGLJ1745.5+1018 2.65 23.9 1.3 0.73 0.014 BW, ",
           "J1747-4036 Parkes 1FGLJ1747.44036 1.64 153.0 3.4 Isolated ",
           "J1810+1744 GBT-350 1FGLJ1810.3+1741 1.66 39.6 2.0 0.15 0.045 BW, ",
           "J1816+4510 GBT-350 1FGLJ1816.7+4509 3.19 38.9 2.4 0.36 0.16 y, RB, ",
           "J1828+0625 GMRT-325 1FGLJ1830.1+0618 3.63 22.4 1.2 6.0 TBD",
           "J1858-2216 GBT-820 1FGLJ1858.12218 2.38 26.6 0.9 46.1 0.22 ",
           "J1902-5105 Parkes 1FGLJ1902.05110 1.74 36.3 1.2 2.01 0.16 ",
           "J1902-70 Parkes 2FGLJ1902.77053 3.60 19.5 0.8 TBD",
           "J2017+0603 Nancay 1FGLJ2017.3+0603 2.90 23.9 1.6 2.2 0.18 ",
           "J2043+1711 Nancay 1FGLJ2043.2+1709 2.38 20.7 1.8 1.48 0.17 ",
           "J2047+1053 GBT-820 1FGLJ2047.6+1055 4.29 34.6 2.0 0.12 0.036 BW, ",
           "J2129-0429 GBT-350 1FGLJ2129.80427 7.62 16.9 0.9 0.64 0.37 RB [16]",
           "J2214+3000 GBT-820 1FGLJ2214.8+3002 3.12 22.5 1.5 0.42 0.014 BW, ",
           "J2215+5135 GBT-350 1FGLJ2216.1+5139 2.61 69.2 3.0 0.17 0.21 RB, ",
           "J2234+0944 Parkes 1FGLJ2234.8+0944 3.63 17.8 1.0 0.42 0.015 BW, ",
           "J2241-5236 Parkes 1FGLJ2241.95236 2.19 11.5 0.5 0.14 0.012 BW, ",
           "J2302+4442 Nancay 1FGLJ2302.8+4443 5.19 13.8 1.2 125.9 0.3 ",
           ]
    pulsars = {}
    for row in lst:
        name = row.split()[0]
        p0 = float(row.split()[3])/1000.  #[ms] --> [s]
        dm = float(row.split()[4])
        coords = name.strip('J')
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
        else:
            raj = coords.split('-')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))

        pulsars[name] =  pulsar(name=name, psrj=name,
                                ra=raj, dec=decj,
                                P0=p0, DM=dm, gbncc=True,
                                catalog=url)
    return pulsars
        
        
def GBT350NGP_pulsarlist():
    """
    pulsars found in:
    http://arxiv.org/pdf/0710.1745v1.pdf
    name, P[ms], DM
    """
    url = 'http://arxiv.org/pdf/0710.1745v1.pdf'
    lst =  ['J0033+57 315 76',
            'J0033+61 912 37',
            'J0054+66 1390 15',
            'J0058+6125 637 129', 
            'J0240+62 592 4 ',
            'J0243+6027 1473 141',
            'J0341+5711 1888 100',
            'J0408+55A 1837 55 ',
            'J0408+55B 754 64 ',
            'J0413+58 687 57 ',
            'J0419+44 1241 71 ',
            'J0426+4933 922 85',
            'J0519+44 515 52',
            'J2024+48 1262 99',
            'J2029+45 1099 228',
            'J2030+55 579 60',
            'J2038+35 160 58',
            'J2043+7045 588 57',
            'J2102+38 1190 85',
            'J2111+40 4061 120 ',
            'J2138+4911 696 168',
            'J2203+50 745 79',
            'J2208+5500 933 105',
            'J2213+53 751 161',
            'J2217+5733 1057 130',
            'J2222+5602 1336 168',
            'J2238+6021 3070 182',
            'J2244+63 461 92',
            'J2315+58 1061 74',
            'J2316+64 216 248',
            'J2326+6141 790 33',
            'J2343+6221 1799 117',
            'J2352+65 1164 152',
            ]
    pulsars = {}
    for row in lst:
        name, p0, dm = row.split()
        p0 = float(p0)/1000. #[ms] --> [s]
        dm = float(dm)
        coords = name.strip('J')
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
        else:
            raj = coords.split('-')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
        pulsars[name] =  pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=True,
                                catalog=url)
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
    def __init__(self, name, psrj, ra, dec, P0, DM, gbncc=False, catalog=None):
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
        self.catalog = catalog

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
            deg = (float(s[0]) + float(s[1])/60.))*360./24.
        elif ncomp >= 3:
            deg = (float(s[0]) + float(s[1])/60. + float(s[2])/3600.)*360./24.
    else:
        if len(hhmm) == 2:
            deg = float(hhmm)*360./24.
        elif len(hhmm) == 4:
            deg = (float(hhmm[0:2]) + float(hhmm[2:4])/60.) *360./24.
        elif len(hhmm) >= 6:
            deg = (float(hhmm[0:2]) + float(hhmm[2:4])/60.\
                + float(hhmm[4:6])/3600.)*360./24.
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
        if v.gbncc or pulsar.gbncc:
            if abs(ra - pra) <= sep and \
                    abs(dec - pdec) <= 1.2:
                matches[k] = v
        elif abs(ra - pra) <= sep and \
                abs(dec - pdec) <= sep:
            matches[k] = v
    return sorted(matches.values(), key=lambda x: x.name, reverse=True) #gives a sorted list
    
        
    

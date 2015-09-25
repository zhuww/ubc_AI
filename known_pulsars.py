"""
Aaron Berndsen: a module which reads the ATNF database and 
GBNCC discoveries page and provides a list of known pulsars.

Pulsars are given as ephem.FixedBody objects, with all the 
ATNF pulsar properties in self.props[key].

"""

import numpy as np
import re
import urllib

#from BeautifulSoup import BeautifulSoup

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
                'lofar' : LOFAR_pulsarlist(),
                'ryan': ryan_pulsars(),
                }

    for k, v in atnf.iteritems():
        allpulsars[k] = v
    for survey, lst in two_join.iteritems():
        for k, v in lst.iteritems():
            if k in allpulsars:
                #add it in if P0 differ by 10ms
                try:
                    if int(100*allpulsars[k].P0) == int(100*v.P0):
                        print "pulsar %s from %s already in our database" % (k, survey)
                        continue
                    else:
                        k += 'a'
                except(ValueError):
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
    pulsars = {}
    url = 'http://arcc.phys.utb.edu/gbncc/'
    try:
        from BeautifulSoup import BeautifulSoup
        sock = urllib.urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data)
        sock.close()
        rows = soup.findAll('tr')[1:]
    except(IOError):
        rows = []
    for row in rows:
        cols = row.findAll('td')
        name = cols[1].text
        p0 = float(cols[2].text)/1000. #ms --> [s]
        dm = cols[3].text
        coords = name.strip('J')   
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
                gbncc = True
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        else:
            raj = coords.split('')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
                gbncc = True
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
                gbncc = False

        pulsars[name] = pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=gbncc,
                               catalog=url)

    return pulsars

def LOFAR_pulsarlist():
    """
    gather the pulsars listed on the LOFAR lotas discovery page
    """
    pulsars = {}
    url = 'http://astron.nl/pulsars/lofar/surveys/lotas/'
    try:
        sock = urllib.urlopen(url)
        data = sock.read()
        sock.close()
    except(IOError):
        data = ''
    datas = data.splitlines()
    #read until '-----------'
    for n, l in enumerate(datas):
        try:
            if l[0] != 'J': continue
            ldata = l.split()
            name = ldata[0]
            DM = ldata[1]
            P0 = ldata[2]
            ra = ldata[5]
            dec = ldata[6]
            pulsars[name] = pulsar(name=name, psrj=name,
                                   ra=ra, dec=dec,
                                   P0=P0, DM=DM,
                                   catalog='lofar'
                                   )
        except(IndexError):
            pass

    return pulsars

def PALFA_pulsarlist():
    """
    gather the pulsars listed on the palfa discovery page
    http://www.naic.edu/~palfa/newpulsars/

    return:
    dictionary of pulsars keyed by PSRJ
    """
    pulsars = {}
    url = 'http://www.naic.edu/~palfa/newpulsars/'
    try:
        from BeautifulSoup import BeautifulSoup
        sock = urllib.urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data)
        sock.close()
        table = soup.findAll('table')[0] #pulsars are in first table
        rows = table.findAll('tr')[1:]
    except(IOError):
        rows = []
    pinfo = PALFA_jodrell_extrainfo()
    for row in rows:
        cols = row.findAll('td')
        name = cols[1].text
#        if name == 'J2010+31': continue #skip single-pulse 
        try:
            p0 = np.float(cols[2].text.strip('~'))/1000. #[ms]-->[s]
        except:
            p0 = np.nan
        if name in pinfo:
            if int(pinfo[name][0]*1000) == int(p0*1000):
                dm = pinfo[name][1]
            else:
                dm = np.nan
        else:
            # the website has updated to include the DM :)
            try:
                dm = np.float(cols[3].text.strip())
            except:
                dm = np.nan 
        coords = name.strip('J')
        if '+' in coords:
            raj = coords.split('+')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('+')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('+%s:00' % match)
                gbncc = True
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        else:
            raj = coords.split('')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
                gbncc = True
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
                gbncc = False

        pulsars[name] = pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=gbncc,
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
    pulsars = {}
    url = 'http://www.jodrellbank.manchester.ac.uk/research/pulsar/PALFA/'
    try:
        from BeautifulSoup import BeautifulSoup
        sock = urllib.urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data)
        sock.close()
        table = soup.findAll('table')[0] #pulsars are in first table
        rows = table.findAll('tr')[1:]
    except(IOError):
        rows = []
    for row in rows:
        cols = row.findAll('td')
        name = 'J%s' % cols[0].text.strip('_jb.tim')
        dm = np.float(cols[3].text)
        p0 = np.float(cols[2].text)/1000. #[ms] --> [s]
        coords = name.strip('J')
        pulsars[name] = (p0, dm)
    return pulsars


def driftscan_pulsarlist():
    pulsars = {}
# moved url = 'http://www.as.wvu.edu/~pulsar/GBTdrift350/'
    url = 'http://astro.phys.wvu.edu/GBTdrift350/'
    try:
        from BeautifulSoup import BeautifulSoup
        sock = urllib.urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data)
        sock.close()
        table = soup.findAll('table')[0] #pulsars are in first table
        rows = table.findAll('tr')
    except(IOError, IndexError):
        rows = []

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
                gbncc = True
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        else:
            raj = coords.split('-')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
                gbncc = True
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
                gbncc = False

        pulsars[name] =  pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=gbncc,
                                catalog=url)
    return pulsars

def ao327_pulsarlist():
    pulsars = {}
    url = 'http://www.naic.edu/~deneva/drift-search/'
    try:
        from BeautifulSoup import BeautifulSoup
        sock = urllib.urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data)
        sock.close()
        table = soup.findAll('table')[0] #pulsars are in first table
        rows = table.findAll('tr')[1:]
    except(IOError, IndexError):
        rows = []

    for row in rows:
        cols = row.findAll('td')
        name = str(cols[1].text).strip('*')
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
                gbncc = True
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        else:
            raj = coords.split('-')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
                gbncc = True
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
                gbncc = False

        pulsars[name] =  pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=gbncc,
                                catalog=url)
    return pulsars

def deepmb_pulsarlist():
    pulsars = {}
    url = 'http://astro.phys.wvu.edu/dmb/'
    try:
        from BeautifulSoup import BeautifulSoup
        sock = urllib.urlopen(url)
        data = sock.read()
        soup = BeautifulSoup(data)
        sock.close()
        table = soup.findAll('table')[0] #pulsars are in first table
        rows = table.findAll('tr')[1:]
    except(IOError, IndexError):
        rows = []
    for row in rows:
        cols = row.findAll('td')
        name = str(cols[0].text)
        if not name.startswith('J'):
            name = 'J%s' % name
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
                gbncc = True
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        else:
            raj = coords.split('-r')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
                gbncc = True
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        pulsars[name] =  pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=gbncc,
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
           ###
    lst2 = [
           "J0023+0923   00:23:30      +09:23:00       0.003050     14.300    MSP   Fermi",
           "J0102+4839   01:02:00      +48:39:00       0.002960     53.500    MSP   Fermi",
           "J0307+7443   03:07:00      +74:43:00       0.003160      6.400    MSP   Fermi",
           "J0340+4130   03:40:24      +41:30:00       0.003300     49.600    MSP   Fermi",
           "J0533+67     05:33:54      +67:59:00       0.004390     57.400    MSP   Fermi",
           "J0605+37     06:05:18      +37:58:00       0.002730     21.000    MSP   Fermi",
           "J0621+25     06:21:12      +25:08:00       0.002720     83.600    MSP   Fermi",
           "J1124-3653   11:24:24      -36:53:00       0.002410     44.900    MSP   Fermi",
           "J1142+0119   11:42:42      +01:19:00       0.005070     19.200    MSP   Fermi",
           "J1207-5050   12:07:00      -50:50:00       0.004840     50.600    MSP   Fermi",
           "J1301+0833   13:01:48      +08:33:00       0.001840     13.200    MSP   Fermi",
           "J1312+0051   13:12:36      +00:51:00       0.004230     15.300    MSP   Fermi",
           "J1514-4946   15:14:06      -49:46:00       0.003590     30.900    MSP   Fermi",
           "J1536-4948   15:36:30      -49:48:00       0.003080     38.000    MSP   Fermi",
           "J1544+4937   15:44:00      +49:37:00       0.002160     23.200    MSP   Fermi",
           "J1551-06     15:51:00      -06:59:00       0.007090     21.600    MSP   Fermi",
           "J1628-3205   16:27:48      -32:05:00       0.003210     42.100    MSP   Fermi",
           "J1630+37     16:30:18      +37:32:00       0.003320     14.100    MSP   Fermi",
           "J1646-2142   16:46:00      -21:42:00       0.005850     29.800    MSP   Fermi",
           "J1658-5324   16:58:48      -53:24:00       0.002430     30.800    MSP   Fermi",
           "J1745+1017   17:45:30      +10:17:00       0.002650     23.900    MSP   Fermi",
           "J1747-4036   17:47:24      -40:36:00       0.001640    153.000    MSP   Fermi",
           "J1810+1744   18:10:18      +17:44:00       0.001660     39.600    MSP   Fermi",
           "J1816+4510   18:16:42      +45:10:00       0.003190     38.900    MSP   Fermi",
           "J1828+0625   18:28:00      +06:25:00       0.003630     22.400    MSP   Fermi",
           "J1858-2216   18:58:06      -22:16:00       0.002380     26.600    MSP   Fermi",
           "J1902-5105   19:02:00      -51:05:00       0.001740     36.300    MSP   Fermi",
           "J1902-70     19:02:42      -70:53:00       0.003600     19.500    MSP   Fermi",
           "J2047+1053   20:47:36      +10:53:00       0.004290     34.600    MSP   Fermi",
           "J2129-0429   21:29:48      -04:29:00       0.007620     16.900    MSP   Fermi",
           "J2215+5135   22:15:00      +51:35:00       0.002610     69.200    MSP   Fermi",
           "J2234+0944   22:34:48      +09:44:00       0.003630     17.800    MSP   Fermi",
           "J1400-56     14:00:00      -56:00:00       0.410700    123.000   YPSR   Fermi",
           "J1203-62     12:03:00      -62:00:00       0.393090    285.000   YPSR   Fermi",
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
                gbncc = True
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        else:
            raj = coords.split('-')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
                gbncc = True
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
                gbncc = False

        pulsars[name] =  pulsar(name=name, psrj=name,
                                ra=raj, dec=decj,
                                P0=p0, DM=dm, gbncc=gbncc,
                                catalog=url)

    for row in lst2:
        name = row.split()[0]
        p0 = float(row.split()[3])
        dm = float(row.split()[4])
        ra = row.split()[1]
        dec = row.split()[2]
        pulsars[name] = pulsar(name=name, psrj=name,
                               ra=ra, dec=dec, P0=p0,
                               DM=dm, gbncc=False,
                               catalog='FERMI')
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
                gbncc = True
            else:
                decj = str('+%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        else:
            raj = coords.split('-')[0]
            raj = str('%s:%s' % (raj[0:2],raj[2:]))
            tmp = coords.split('-')[1]
            match = re.match('\d+',tmp).group(0)
            if len(match) == 2:
                decj = str('-%s:00' % match)
                gbncc = True
            else:
                decj = str('-%s:%s' % (match[0:2],match[2:4]))
                gbncc = False
        pulsars[name] =  pulsar(name=name, psrj=name,
                               ra=raj, dec=decj,
                               P0=p0, DM=dm, gbncc=gbncc,
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
    deg = 0.0
    if ':' in hhmm:
        s = hhmm.split(':')
        ncomp = len(s)
        if ncomp == 1:
            deg = float(s[0])*360./24.
        elif ncomp == 2:
            deg = (float(s[0]) + float(s[1])/60.)*360./24.
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
        elif len(ddmm) == 3: # matches +?? or -??
            deg = abs(float(ddmm))
        elif len(ddmm) == 4:
            deg = abs(float(ddmm[0:2])) + float(ddmm[2:4])/60.
        elif len(ddmm) >= 6:
            deg = abs(float(ddmm[0:2])) + float(ddmm[2:4])/60.\
                + float(ddmm[4:6])/3600.
    return deg*sgn
    
def matches(allpulsars, pulsar, sep=.6, harm_match=False, DM_match=False):
    """
    given a dictionary of all pulsars and a pulsar object,
    return the objects within 'sep' degrees of the pulsar.

    args:
    allpulsars : dictionary of allpulsars
    pulsar : object of interest
    sep : degrees of separation [default .6 deg]

    Optional:
    harmonic_match : [default False], if True reject match if it is
              not a harmonic ratio 
              (we print the rejected object so you can follow up)
    DM_match : [default False], if True reject match if it is there
               is a 15% difference in DM
    

    Notes:
    *because GBNCC only gives Dec to nearest degree, we consider
    things close if they are separated by less than a degree in dec,
    and 'sep' in RA
    *if the known pulsar is a 'B' pulsar, we use a sep of max(sep, 5degrees)
     
    """
    matches = {}
    pra = hhmm2deg(pulsar.ra)
    pdec = ddmm2deg(pulsar.dec)
    orig_sep = sep
    for k, v in allpulsars.iteritems():
        amatch = False

        ## find positional matches
        ra = hhmm2deg(v.ra)
        dec = ddmm2deg(v.dec)
        #use very wide "beam" for bright pulsars (the "B" pulsars)
        if v.name.startswith('B'):
            sep = max(orig_sep, 2.5)
        else:
            sep = orig_sep
        dra = abs(ra - pra)*np.cos(pdec*np.pi/180.)
        ddec = abs(dec - pdec)
        if v.gbncc or pulsar.gbncc:
            #make sure dec is at least one degree
            if dra <= sep and ddec <= max(1.2,sep):
                amatch = True
        elif dra <= sep and ddec <= sep:
            amatch = True
            
        ## reject nearby objects if they aren't harmonics
        if amatch and harm_match:
            max_denom = 100
            num, den = harm_ratio(np.round(pulsar.P0,5), np.round(v.P0,5), 
                                  max_denom=max_denom)
            if num == 0:
                num = 1
                den = max_denom
            pdiff = abs(1. - float(den)/float(num) *pulsar.P0/v.P0)
            if pdiff > 1:
                amatch = False
                print "%s is not a harmonic match (rejecting)" % k

        ## reject nearby objects if 15% difference in DM
        if amatch and DM_match:
            if (v.DM != np.nan) and (pulsar.DM != 0.):
                dDM = abs(v.DM - pulsar.DM)/pulsar.DM
                if dDM > 0.26: #0.15:
                    amatch = False
                    print "%s has a very different DM (rejecting)" % k

        ## finally, we've passed location, harmonic and DM matching
        if amatch:
            matches[v.name] = v
    return sorted(matches.values(), key=lambda x: x.name, reverse=True) #gives a sorted list
    
def harm_ratio(a,b,max_denom=100):
    """
    given two numbers, find the harmonic ratio

    """
    c = fractions.Fraction(a/b).limit_denominator(max_denominator=max_denom)
    return c.numerator, c.denominator

      
    
def ryan_pulsars():
    """
    return the list of pulsars that Ryan Lynch provided

    """
    pulsars = {}
    for row in ryans_list.splitlines():
        if not row: continue
        if row.startswith('#'): continue
        try:
            name, ra, dec, p0, dm, typ, catalog = row.split()
        except(ValueError):
            continue
        if len(dec.split(':')) == 2 or '00:00' in dec:
            #Dec not listed accurately
            gbncc = True
        else:
            gbncc = False
        pulsars[name] = pulsar(name=name, psrj=name,
                         ra=ra, dec=dec,
                         P0=p0, DM=dm, gbncc=gbncc,
                         catalog=catalog)
    return pulsars
    
ryans_list = """# New GBNCC pulsars
J0636+51     06:36:59      +51:29:42       0.002869     11.110    MSP   GBNCC 
J0645+51     06:45:58      +51:50:06       0.008535     18.242    MSP   GBNCC 
J0740+41     07:40:45      +41:03:53       0.003139     20.833    MSP   GBNCC 
J0741+66     07:41:59      +66:20:11       0.002886     14.960    MSP   GBNCC 
J1124+78     11:24:18.59   +78:22:22.6:49  0.004201     11.220    MSP   GBNCC 
J1649+80     16:49:54      +80:44:49       0.002021     31.090    MSP   GBNCC 
J1710+49     17:10:29      +49:19:51       0.003220      7.087    MSP   GBNCC 
J1816+45     18:16:36      +45:10:34       0.003193     38.896    MSP   GBNCC 
J1953+67     19:53:12      +67:01:46       0.008565     57.161    MSP   GBNCC 
J0125+62     01:25:42      +62:35:55       1.708233    117.688    NRP   GBNCC
J0216+52     02:16:19      +52:13:52       0.024576     22.050    ???   GBNCC 
J0357+66     03:57:32      +66:40:06       0.091507     62.310    ???   GBNCC 
J0510+38     05:09:59      +38:12:11       0.076564     69.226    ???   GBNCC 
J1432+72     14:32:32      +72:36:06       0.041741     12.586    ???   GBNCC 
J1939+66     19:39:49      +66:11:38       0.022261     41.221    ???   GBNCC 
J0053+69     00:53:24      +69:38:45       0.832903    116.679    NRP   GBNCC 
J0059+50     00:59:12      +50:01:58       0.996009     67.008    NRP   GBNCC 
J0112+66     01:12:37      +66:21:40       4.301240    112.127    NRP   GBNCC 
J0136+63     01:36:19      +63:42:22       0.717895    285.939    NRP   GBNCC 
J0213+52     02:13:22      +52:31:38       0.376384     37.925    NRP   GBNCC 
J0325+67     03:25:54      +67:49:10       1.364741     65.268    NRP   GBNCC 
J0338+66     03:38:35      +66:44:04       1.762000     66.561    NRP   GBNCC 
J0359+42     03:59:36      +41:59:56       0.226478     46.240    NRP   GBNCC 
J0519+54     05:19:48      +54:25:18       0.340194     42.598    NRP   GBNCC 
J0610+37     06:10:39      +37:18:14       0.443861     38.738    NRP   GBNCC 
J0614+83     06:14:03      +83:13:46       1.039203     44.134    NRP   GBNCC 
J0645+80     06:45:32      +80:08:54       0.657873     49.723    NRP   GBNCC
J0737+69     07:37:10      +69:13:46       6.824240     16.288    NRP   GBNCC 
J0746+66     07:46:28      +66:36:29       0.407670     27.785    NRP   GBNCC 
J0749+57     07:49:56      +57:00:10       1.174875     26.610    NRP   GBNCC
J0943+41     09:43:22      +41:08:57       2.229489     21.286    NRP   GBNCC 
J1101+65     11:01:34      +65:07:06       3.631320     18.504    NRP   GBNCC 
J1110+58     11:10:41      +58:52:00       0.793348     26.406    NRP   GBNCC
J1320+67     13:20:58      +67:30:18       1.028620     27.981    NRP   GBNCC
J1627+86     16:27:02      +86:54:29       0.395785     46.474    NRP   GBNCC 
J1629+43     16:29:20      +43:58:57       0.181173      7.342    NRP   GBNCC 
J1647+66     16:47:38      +66:03:57       1.599862     22.739    NRP   GBNCC 
J1706+59     17:06:44      +59:09:36       1.476687     30.596    NRP   GBNCC
J1800+50     18:00:44      +50:28:26       0.578364     22.550    NRP   GBNCC 
J1815+55     18:15:03      +55:28:40       0.426802     58.800    NRP   GBNCC 
J1821+41     18:21:41      +41:44:34       1.261787     40.312    NRP   GBNCC 
J1859+76     18:59:01      +76:53:34       1.393617     47.140    NRP   GBNCC 
J1921+42     19:21:56      +42:24:39       0.595186     53.184    NRP   GBNCC 
J1922+58     19:22:05      +58:27:59       0.529623     53.250    NRP   GBNCC
J1935+52     19:35:05      +52:11:33       0.568387     71.065    NRP   GBNCC 
J1941+43     19:41:32      +43:23:01       0.840801     79.440    NRP   GBNCC 
J1942+81     19:42:03      +81:06:03       0.203558     40.189    NRP   GBNCC 
J1954+43     19:54:58      +43:50:08       1.386961    130.183    NRP   GBNCC 
J2001+42     20:01:30      +42:43:19       0.719161     54.950    NRP   GBNCC 
J2017+59     20:17:53      +59:12:52       0.403169     60.840    NRP   GBNCC
J2027+74     20:27:34      +74:46:32       0.515229     11.064    NRP   GBNCC 
J2122+54     21:22:46      +54:32:43       0.138866     31.709    NRP   GBNCC 
J2137+64     21:37:20      +64:19:03       1.750870    105.640    NRP   GBNCC 
J2205+62     22:05:38      +62:04:38       0.322787    167.200    NRP   GBNCC
J2207+40     22:07:17      +40:57:24       0.636985     11.607    NRP   GBNCC 
J2228+64     22:28:38      +64:57:50       1.893120    194.311    NRP   GBNCC 
J2243+69     22:43:34      +69:39:34       0.855374     68.028    NRP   GBNCC 
J2316+69     23:16:54      +69:12:03       0.813413     71.160    NRP   GBNCC 
J2353+85     23:53:30      +85:34:14       1.011759     38.319    NRP   GBNCC 

# Soon-to-be published GBT 350 MHz Drift-scan pulsars
J0348+0432   03:48:43.63   +04:32:11.45    0.039123     40.569    PRP   Drift 
J0458-0505   04:58:37.12   -05:05:05.13    1.883480     47.806    NRP   Drift 
J2033+0042   20:33:31.11   +00:42:21.96    5.013398     37.841    NRP   Drift 
J1501-0046   15:01:44.95   -00:46:23.52    0.464037     22.258    NRP   Drift 
J1518-0627   15:18:59.11   -06:27:07.69    0.794997     27.963    NRP   Drift 
J1547-0944   15:47:46.05   -09:44:07.80    1.576925     37.416    NRP   Drift 
J1853-0649   18:53:25.42   -06:49:25.94    1.048132     44.541    NRP   Drift 
J1918-1052   19:18:48.24   -10:52:46.37    0.798693     62.731    NRP   Drift 
J2013-0649   20:13:17.75   -06:49:05.39    0.580187     63.368    NRP   Drift 
J1923+2515   19:23:22.49   +25:15:40.64    0.003788     18.858    MSP   Drift 
J1327-0755   13:27:57.58   -07:55:29.80    0.002678     27.895    MSP   Drift 
J1555-0515   15:55:40.10   -05:15:56.84    0.975410     23.489    NRP   Drift 
J1612+2008   16:12:23.42   +20:08:18.37    0.426646     19.539    NRP   Drift 
J1623-0841   16:23:42.71   -08:41:36.37    0.503015     60.433    NRP   Drift 
J1633-2009   16:33:55.34   -20:09:58.55    0.935557     48.201    NRP   Drift 
J1735-0243   17:35:48.30   -02:43:51.24    0.782887     55.321    NRP   Drift 
J1737-0811   17:37:47.11   -08:11:08.88    0.004175     55.350    MSP   Drift 
J1903-0848   19:03:11.26   -08:48:56.78    0.887325     66.987    NRP   Drift 
J1941+0121   19:41:16.04   +01:21:39.50    0.217317     52.646    NRP   Drift 
J2012-2029   20:12:46.75   -20:29:42.18    0.544002     37.632    NRP   Drift 
J2033-1938   20:33:54.76   -19:38:27.03    1.281719     23.469    NRP   Drift 
J2111+2106   21:11:33.12   +21:06:07.01    3.953853     59.772    NRP   Drift 
J2222-0137   22:22:05.96   -01:37:15.72    0.032818      3.278    PRP   Drift 
J2256-1024   22:56:45.00   -10:24:37.00    0.002295     13.777    MSP   Drift 
# Already published GBT 350 MHz Drift-scan pulsars
#J1023+0038   10:23:47.68   +00:38:40.85    0.001688     14.325    MSP   Drift
#New GBT 350 MHz Drift-scan pulsars
J0337+17     03:37:37.50   +17:14:31.00    0.002733     21.320    NRP   Drift 
J0931-19     09:31:28      -19:05:11       0.004638     41.483    MSP   Drift 
J1930-01     19:30:05      -01:53:32       0.593683     35.959    NRP   Drift 
J1745-01     17:45:47      -01:03:45       0.679534     67.459    NRP   Drift 
J1758-10     17:58:52      -10:15:57       2.512740    119.740    NRP   Drift 
J1132+24     11:32:09      +24:55:59       0.507072     23.269    NRP   Drift 
J1444+18     14:44:08      +18:00:20       0.132518     16.988    NRP   Drift 
J1543-07     15:43:45      -07:05:36       0.242063     30.378    NRP   Drift 
J1643-10     16:43:02      -10:15:09       0.062716     76.004    ???   Drift 
# New GBT 350 MHz Drift-scan RRATs
J1704-04     17:04:54      -04:40:38       0.238000     43.000   RRAT   Drift 
J1914-11     19:14:46      -11:29:34              *     90.000   RRAT   Drift
J1537+23     15:37:47      +23:50:51       3.450000     15.000   RRAT   Drift 
J1944-11     19:44:09      -10:17:07       0.409000     31.000   RRAT   Drift 
J2006+20     20:06:55      +20:06:55       4.630000     66.000   RRAT   Drift 
J2324-05     23:24:22      -05:07:36       0.869000     15.000   RRAT   Drift 

# New GBT North Galactic Plane pulsars
J0033+57     00:33:00      +57:00:00       0.315000     76.000    NRP   GBTNGP
J0033+61     00:33:00      +61:00:00       0.912000     37.000    NRP   GBTNGP
J0054+66     00:54:00      +66:00:00       1.390000     15.000    NRP   GBTNGP
J0058+6125   00:58:00      +61:25:00       0.637000    129.000    NRP   GBTNGP
J0240+62     02:40:00      +62:00:00       0.592000      4.000    NRP   GBTNGP
J0243+6027   02:43:00      +60:27:00       1.473000    141.000    NRP   GBTNGP
J0341+5711   03:41:00      +57:11:00       1.888000    100.000    NRP   GBTNGP
J0408+55A    04:08:00      +55:00:00       1.837000     55.000    NRP   GBTNGP
J0408+55B    04:08:00      +55:00:00       0.754000     64.000    NRP   GBTNGP
J0413+58     04:13:00      +58:00:00       0.687000     57.000    NRP   GBTNGP
J0419+44     04:19:00      +44:00:00       1.241000     71.000    NRP   GBTNGP
J0426+4933   04:26:00      +49:33:00       0.922000     85.000    NRP   GBTNGP
J0519+44     05:19:00      +44:00:00       0.515000     52.000    NRP   GBTNGP
J2024+48     20:24:00      +48:00:00       1.262000     99.000    NRP   GBTNGP
J2029+45     20:29:00      +45:00:00       1.099000    228.000    NRP   GBTNGP
J2030+55     20:30:00      +55:00:00       0.579000     60.000    NRP   GBTNGP
J2038+35     20:38:00      +35:00:00       0.160000     58.000    NRP   GBTNGP
J2043+7045   20:43:00      +70:45:00       0.588000     57.000    NRP   GBTNGP
J2102+38     21:02:00      +38:00:00       1.190000     85.000    NRP   GBTNGP
J2111+40     21:11:00      +40:00:00       4.061000    120.000    NRP   GBTNGP
J2138+4911   21:38:00      +49:11:00       0.696000    168.000    NRP   GBTNGP
J2203+50     22:03:00      +50:00:00       0.745000     79.000    NRP   GBTNGP
J2208+5500   22:08:00      +55:00:00       0.933000    105.000    NRP   GBTNGP
J2213+53     22:13:00      +53:00:00       0.751000    161.000    NRP   GBTNGP
J2217+5733   22:17:00      +57:33:00       1.057000    130.000    NRP   GBTNGP
J2222+5602   22:22:00      +56:02:00       1.336000    168.000    NRP   GBTNGP
J2238+6021   22:38:00      +60:21:00       3.070000    182.000    NRP   GBTNGP
J2244+63     22:44:00      +63:00:00       0.461000     92.000    NRP   GBTNGP
J2315+58     23:15:00      +58:00:00       1.061000     74.000    NRP   GBTNGP
J2316+64     23:16:00      +64:00:00       0.216000    248.000    NRP   GBTNGP
J2326+6141   23:26:00      +61:41:00       0.790000     33.000    NRP   GBTNGP
J2343+6221   23:43:00      +62:21:00       1.799000    117.000    NRP   GBTNGP
J2352+65     23:52:00      +65:00:00       1.164000    152.000    NRP   GBTNGP
# New PALFA pulsars
J0557+15     05:57:34.90   +15:51:28.10    0.002556    102.655    MSP   PALFA 
J1851+0242   18:51:22      +02:42:37       1.497140    534.100    NRP   PALFA 
J1855+03     18:55:41.00   +03:07:07.10    1.633460    625.600    NRP   PALFA 
J1857+02     18:57:13.10   +02:59:46.40    0.772747    681.387    NRP   PALFA 
J1858+0319   18:58:39.50   +03:19:01.90    0.867378    289.827    NRP   PALFA 
J1900+04     19:00:14      +04:39:08       0.312302    638.958    NRP   PALFA 
J1901+0510   19:01:48      +05:10:20       0.657154    417.900    NRP   PALFA 
J1903+06     19:03:52.70   +06:54:09.30    0.791215    353.302    NRP   PALFA 
J1905+04     19:05:00.70   +04:53:05.00    0.006092    182.859    MSP   PALFA 
J1907+09     19:07:02.90   +09:00:47.90    1.527060    189.864    NRP   PALFA 
J19088+08    19:08:18.30   +08:31:43.90    0.512203    698.100    NRP   PALFA 
J1912+09     19:12:02.90   +09:24:51.40    0.323839    490.950    NRP   PALFA 
J1913+10     19:13:31.90   +10:50:12.80    0.190666    230.048    NRP   PALFA 
J1913+1102   19:13:35      +11:02:56       0.027295    338.218    ???   PALFA 
J1914+14     19:14:56.30   +14:28:13.60    1.159482    206.035    NRP   PALFA 
J1918+13     19:18:44.10   +13:10:16.40    0.856741    247.412    NRP   PALFA 
J1922+11     19:22:48.90   +11:33:47.20    0.562065    331.295    NRP   PALFA 
J1929+11     19:29:20.70   +11:57:24.2     3.218000     80.000   RRAT   PALFA 
J1933+17     19:33:20.80   +17:28:06.50    0.021510    156.600    ???   PALFA 
J1938+2013   19:38:42.50   +20:12:56.70    0.002634    237.088    MSP   PALFA 
J1953+24     19:53:59      +24:07:44       0.193406     80.825    NRP   PALFA 
J2005+26     20:05:01.90   +26:55:23.70    0.665846    158.548    NRP   PALFA 
J1853+03     18:53:12.50   +02:59:59.80    0.585533    290.200    NRP   PALFA 
J1854+0320   18:54:05.20   +03:20:42.50    0.628499    479.482    NRP   PALFA 
J1901+05     19:01:22.20   +04:59:58.10    0.877073   1103.561    NRP   PALFA 
J1902+03     19:02:05.10   +03:00:48.60    0.007796    254.087    MSP   PALFA 
J1906+01     19:06:46.80   +00:55:03.60    0.002789    127.000    MSP   PALFA 
J1906+05     19:06:52.90   +05:10:16.10    0.397588    104.561    NRP   PALFA 
J1907+02     19:07:11      +02:56:18       0.618789    250.403    NRP   PALFA 
J1907+06     19:07:09.90   +06:30:13.10    0.323622    427.575   YPSR   PALFA 
J1908+08     19:07:57.10   +08:33:23.50    0.167616    511.532    NRP   PALFA 
J1909+11     19:09:23.10   +11:49:30.90    0.448956    204.452    NRP   PALFA 
J1909+12     19:09:57.70   +12:04:55.30    1.229338    292.481    NRP   PALFA 
J1910+05     19:10:40.50   +05:17:37.80    0.308038    305.853    NRP   PALFA 
J1910+1017   19:10:29.40   +10:17:33.90    0.411089    636.561    NRP   PALFA 
1910+1027    19:10:41.20   +10:27:34.60    0.531490    706.488    NRP   PALFA 
J1911+09     19:11:49.10   +09:23:27.30    0.273706    334.657    NRP   PALFA 
J1913+11     19:13:39.30   +11:03:22.40    0.923910    628.900    NRP   PALFA 
J1914+07     19:14:21.50   +06:59:27.50    0.018512    226.144    MSP   PALFA 
J1915+11     19:15:34      +11:44:31       0.173646    338.251    NRP   PALFA 
J1924+17     19:24:24.80   +17:13:13.90    0.758432    530.715    NRP   PALFA 
J1925+17     19:25:31      +17:21:51       0.075655    223.745    ???   PALFA 
J1930+14     19:30:23.20   +14:08:32.20    0.425713    208.055    NRP   PALFA 
J1931+14     19:31:38.50   +14:38:55.60    0.593075    239.318    NRP   PALFA 
J1934+19     19:34:20.40   +19:26:14.70    0.231012     97.493    NRP   PALFA 
J1940+22     19:40:32.40   +22:44:52.10    0.258897    218.081    NRP   PALFA 
J1950+24     19:50:48.80   +24:16:06.60    0.004305    142.293    MSP   PALFA 
J1952+25     19:52:22.80   +25:12:18.90    1.077668    245.382    NRP   PALFA 
J1957+25     19:57:36.40   +25:15:54.50    0.003961     44.149    MSP   PALFA 
J2010+31     20:10:46.30   +31:46:32.60           *    251.000   RRAT   PALFA
J1858+02     18:58:18.30   +02:39:52.40    0.197623    496.749    NRP   PALFA 
J1944+22     19:44:02.00   +22:34:50.30    0.003618    184.678    MSP   PALFA 
J1844+01     18:44:42.50   +01:15:50.30    0.004185    148.055    MSP   PALFA 
J1850+01     18:50:03.10   +01:24:37.00    0.003560    119.054    NRP   PALFA 
J1859+06     18:59:43.50   +06:04:49.00    0.508564    381.875    NRP   PALFA 
J1900+03     19:00:49.60   +03:08:46.10    0.004909    249.933    MSP   PALFA 
J0611+14     06:11:13.80   +14:34:51.90    0.270328     35.409    NRP   PALFA 
J1854+03     18:54:08.50   +03:04:44.10    4.559620     43.000   RRAT   PALFA 
J1905+09     19:05:33.30   +08:57:30.20    3.487840    288.000   RRAT   PALFA 
J1917+17     19:17:28.10   +17:36:41.00    0.334723    209.579    NRP   PALFA 
J1919+17     19:19:43.34   +17:45:03.86    2.081343    148.400    NRP   PALFA 
J1955+25     19:55:56.30   +25:26:37.70    0.004873    210.000    MSP   PALFA 
J1947+19     19:47:18.30   +19:57:33.50    0.157509    189.672    NRP   PALFA 
J1949+31     19:49:26.00   +31:05:31.80    0.013138    164.161    MSP   PALFA 
J1848+03     18:48:45.00   +03:51:07.50    0.191449    353.300    NRP   PALFA 
J1850+04     18:50:21.30   +04:22:27.00    0.290741    296.800    NRP   PALFA 
J1858+03     18:58:27.20   +03:46:17.80    0.256838    424.000    NRP   PALFA 
J1916+12     19:16:21.00   +12:25:30.90    0.227367    254.400    NRP   PALFA 
J1924+16     19:24:57.20   +16:31:33.00    2.935246    656.300    NRP   PALFA 
J1934+24     19:34:37.40   +23:53:49.50    0.178437    339.200    NRP   PALFA 
J1948+23     19:48:23.30   +23:33:41.90    0.528352    199.000    NRP   PALFA 
J1948+25     19:48:19.30   +25:52:41.40    0.196627    295.618    NRP   PALFA 
J2005+35     20:05:18.90   +35:48:18.50    0.615064    353.300    NRP   PALFA 
J2007+31     20:07:16.00   +31:20:32.00    0.608183    204.900    NRP   PALFA 
J2013+31     20:13:35.40   +31:00:42.50    0.276044    134.300    NRP   PALFA 
J1855+02     18:55:41.40   +02:06:34.90    0.246837    847.900    NRP   PALFA 
J1909+07     19:09:08.00   +07:49:36.00    0.237158    537.000    NRP   PALFA 
J1919+13     19:19:38.20   +13:15:03.80    0.571400    622.000    NRP   PALFA 
J1938+20     19:37:39.60   +20:11:14.30    0.687082    327.394    NRP   PALFA 
J1940+23     19:40:36.20   +23:37:16.50    0.546853    268.500    NRP   PALFA 
J1941+25     19:41:19.60   +25:24:42.40    2.306150    314.290    NRP   PALFA 
J1946+25     19:46:58.30   +25:35:18.90    0.515167    248.800    NRP   PALFA 
J1949+23     19:49:07.30   +23:06:55.50    1.319530    282.600    NRP   PALFA 
J1953+27     19:53:08.30   +27:33:18.50    1.334081    310.900    NRP   PALFA 
J2006+31     20:06:12.60   +31:01:41.20    0.163989    109.500    NRP   PALFA 
J2010+28     20:10:05.40   +28:45:04.60    0.565390     91.900    NRP   PALFA 
J0628+09     06:28:49.00   +09:09:59.00    1.241213    106.000    NRP   PALFA 
# PALFA pulsars already in ATNF
#J0627+16     06:27:13.10   +16:12:32.10    2.180495    113.000    NRP   PALFA
#J1946+24     19:46:00.60   +23:58:34.30    4.729340     96.000   RRAT   PALFA
#J1909+0640   19:09:23.60   +06:40:41.00    0.741728     34.540    NRP   PALFA
#J1903+03     19:03:11.50   +03:28:44.70    0.002150    297.583    MSP   PALFA
#J1856+02     18:56:54.50   +02:45:09.80    0.080894    621.800   YPSR   PALFA
#J1952+26     19:52:04.20   +26:42:01.50    0.020730    316.279    PRP   PALFA
#J1928+15     19:28:20.40   +15:13:22.20    0.402700    242.000   RRAT   PALFA
#J0540+3207   05:40:37.40   +32:07:38.60    0.524190    120.100    NRP   PALFA
#J2007+27     20:07:14.00   +27:24:26.00    0.024497    127.313    PRP   PALFA
#J1904+07     19:04:08.10   +07:38:40.40    0.208971    282.600    NRP   PALFA
#J1905+08     19:05:16      +09:01:22       0.218247    452.200    NRP   PALFA
#J1906+0746   19:06:48.70   +07:46:28.40    0.144003    212.000    DNS   PALFA
#J1928+1746   19:28:42.50   +17:46:26.70    0.068728    173.900   YPSR   PALFA
#J2009+33     20:09:39.10   +33:25:57.70    1.438519    367.400    NRP   PALFA
#J2010+32     20:10:28.10   +32:36:20.60    1.442401    438.100    NRP   PALFA
#J2011+33     20:11:24.80   +33:26:13.60    0.931709    310.900    NRP   PALFA
#J2018+34     20:18:53.30   +34:31:00.10    0.387670    226.100    NRP   PALFA

# New AO 327 MHz Drift Survey pulsars
J0158+21     01:58:29.28   +21:08:11.20    0.505285     19.629    NRP   AO327
J0229+20     02:29:03.26   +20:58:15.96    0.806884     26.539    NRP   AO327
J0244+14     02:44:50.80   +14:27:23.51    2.128094     30.650    NRP   AO327
J0453+16     04:53:33.82   +16:06:33.90    0.045769     30.272    PRP   AO327
J0457+23     04:57:06.25   +23:38:03.15    0.504904     59.019    NRP   AO327
J2204+27     22:04:51.41   +27:02:07.86    0.084704     35.085    ???   AO327
J0848+16     08:48:53.24   +16:43:13.30    0.452365     38.209    NRP   AO327
J0241+16     02:41:45.91   +16:04:42.63    1.545360     15.539    NRP   AO327

# New DMB pulsars
J1934+1726   19:34:00      +17:26:00       0.004020     61.700    MSP   DMB
J1936+2025   19:36:25.8    +20:25:41.4     0.081224    181.700    ???   DMB
J1921+1540   19:21:48.6    +15:40:30.2     0.143583    484.600    NRP   DMB
J1924+1645   19:24:13.1    +16:45:10.9     0.158048    207.700    NRP   DMB
J1938+2216   19:38:07.2    +22:16:57.3     0.166124     90.900    NRP   DMB
J1922+1722   19:22:35.2    +17:22:27.0     0.236180    238.000    NRP   DMB
J1928+1957   19:28:26.2    +19:57:18.5     0.257850    276.900    NRP   DMB
J1926+2021   19:26:21.6    +20:21:46.5     0.299092    246.600    NRP   DMB
J1929+1905   19:29:31.7    +19:05:43.4     0.339243    553.900    NRP   DMB
J1932+1944   19:32:02.3    +19:44:37.1     0.501151    441.400    NRP   DMB
J1929+1621   19:29:17.8    +16:21:47.4     0.529663     20.600    NRP   DMB
J1919+1642   19:19:06.6    +16:42:45.8     0.562833    207.700    NRP   DMB
J1929+2125   19:29:49.1    +21:25:54.5     0.723647     62.700    NRP   DMB
J1928+1923   19:28:53.7    +19:23:12.6     0.817384    493.300    NRP   DMB
J1925+1900   19:25:00      +19:00:00       1.916000    329.000   RRAT   DMB


# New Fermi pulsars
J0023+0923   00:23:30      +09:23:00       0.003050     14.300    MSP   Fermi
J0102+4839   01:02:00      +48:39:00       0.002960     53.500    MSP   Fermi
J0307+7443   03:07:00      +74:43:00       0.003160      6.400    MSP   Fermi
J0340+4130   03:40:24      +41:30:00       0.003300     49.600    MSP   Fermi
J0533+67     05:33:54      +67:59:00       0.004390     57.400    MSP   Fermi
J0605+37     06:05:18      +37:58:00       0.002730     21.000    MSP   Fermi
J0621+25     06:21:12      +25:08:00       0.002720     83.600    MSP   Fermi
J1124-3653   11:24:24      -36:53:00       0.002410     44.900    MSP   Fermi
J1142+0119   11:42:42      +01:19:00       0.005070     19.200    MSP   Fermi
J1207-5050   12:07:00      -50:50:00       0.004840     50.600    MSP   Fermi
J1301+0833   13:01:48      +08:33:00       0.001840     13.200    MSP   Fermi
J1312+0051   13:12:36      +00:51:00       0.004230     15.300    MSP   Fermi
J1514-4946   15:14:06      -49:46:00       0.003590     30.900    MSP   Fermi
J1536-4948   15:36:30      -49:48:00       0.003080     38.000    MSP   Fermi
J1544+4937   15:44:00      +49:37:00       0.002160     23.200    MSP   Fermi
J1551-06     15:51:00      -06:59:00       0.007090     21.600    MSP   Fermi
J1628-3205   16:27:48      -32:05:00       0.003210     42.100    MSP   Fermi
J1630+37     16:30:18      +37:32:00       0.003320     14.100    MSP   Fermi
J1646-2142   16:46:00      -21:42:00       0.005850     29.800    MSP   Fermi
J1658-5324   16:58:48      -53:24:00       0.002430     30.800    MSP   Fermi
J1745+1017   17:45:30      +10:17:00       0.002650     23.900    MSP   Fermi
J1747-4036   17:47:24      -40:36:00       0.001640    153.000    MSP   Fermi
J1810+1744   18:10:18      +17:44:00       0.001660     39.600    MSP   Fermi
J1816+4510   18:16:42      +45:10:00       0.003190     38.900    MSP   Fermi
J1828+0625   18:28:00      +06:25:00       0.003630     22.400    MSP   Fermi
J1858-2216   18:58:06      -22:16:00       0.002380     26.600    MSP   Fermi
J1902-5105   19:02:00      -51:05:00       0.001740     36.300    MSP   Fermi
J1902-70     19:02:42      -70:53:00       0.003600     19.500    MSP   Fermi
J2047+1053   20:47:36      +10:53:00       0.004290     34.600    MSP   Fermi
J2129-0429   21:29:48      -04:29:00       0.007620     16.900    MSP   Fermi
J2215+5135   22:15:00      +51:35:00       0.002610     69.200    MSP   Fermi
J2234+0944   22:34:48      +09:44:00       0.003630     17.800    MSP   Fermi
J1400-56     14:00:00      -56:00:00       0.410700    123.000   YPSR   Fermi
J1203-62     12:03:00      -62:00:00       0.393090    285.000   YPSR   Fermi
# Fermi pulsars already in ATNF
#J0101-6422   01:01:11.1    -64:22:30.1     0.002573      11.93    MSP   Fermi
#J0614-3329   06:14:10.3    -33:29:54.1     0.003149      37.05    MSP   Fermi
#J1103-5403   11:03:33.2    -54:03:43.2     0.003393     103.92    MSP   Fermi
#J1124-36     11:24         -36:00          0.002410      44.90    MSP   Fermi
#J1231-1411   12:31:11.3    -14:11:43.6     0.003684       8.09    MSP   Fermi
#J1302-32     13:02         -32:00          0.003770      26.20    MSP   Fermi
#J1604-44     16:04:32      -44:43          1.400000          *   YPSR   Fermi
#J2017+0603   20:17:22.7    +06:03:05.5     0.002896      23.92    MSP   Fermi
#J2030+3641   20:30:00.2    +36:41:27.1     0.200129     246.00   YPSR   Fermi
#J2043+1711   20:43:20.8    +17:11:28.9     0.002380      20.71    MSP   Fermi
#J2214+3000   22:14:38.8    +30:00:38.2     0.003119      22.56    MSP   Fermi
#J2241-5236   22:41:42.0    -52:36:36.2     0.002187      11.41    MSP   Fermi
#J2302+4442   23:02:46.9    +44:42:22.0     0.005192      13.76    MSP   Fermi
# Galactic center magnetar 
#J1745-2900   17:45:40.0    -29:00:28.1     0.003764      2000     NRP   EFB
"""

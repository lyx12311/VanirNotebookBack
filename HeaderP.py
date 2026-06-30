import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.utils as au
from astropy.io import fits
import astropy.coordinates as coord
from astropy.table import QTable
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import glob
from astropy.table import QTable
import os

home_dic = os.path.expanduser("~")
print('home directory set to:',home_dic)

plt.rcParams.keys()
plt.rc('font', family='serif')
params = {
   'axes.labelsize': 30,
   'axes.linewidth': 1.5,
   'legend.fontsize': 25,
   'legend.frameon': False,
   'lines.linewidth': 2,
   'xtick.direction': 'in',
   'xtick.labelsize': 25,
   'xtick.major.bottom': True,
   'xtick.major.pad': 10,
   'xtick.major.size': 10,
   'xtick.major.width': 1,
   'xtick.minor.bottom': True,
   'xtick.minor.pad': 3.5,
   'xtick.minor.size': 5,
   'xtick.minor.top': True,
   'xtick.minor.visible': True,
   'xtick.minor.width': 1,
   'xtick.top': True,
   'ytick.direction': 'in',
   'ytick.labelsize': 25,
   'ytick.major.pad': 10,
   'ytick.major.size': 10,
   'ytick.major.width': 1,
   'ytick.minor.pad': 3.5,
   'ytick.minor.size': 5,
   'ytick.minor.visible': True,
   'ytick.minor.width': 1,
   'ytick.right': True,
   'figure.figsize': [10,10], # instead of 4.5, 4.5
   'savefig.format': 'eps',
}
plt.rcParams.update(params)

# use to print progress bar
import time, sys
from IPython.display import clear_output
def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1

    block = int(round(bar_length * progress))

    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)
    
# calcualte v_t, v_b by passing in a dataframe with parallax, pmra, pmdec, ra, dec
def CalcV(df):
	d = coord.Distance(parallax=np.array(df.parallax) * u.mas,allow_negative=True)
	vra = (np.array(df.pmra)*u.mas/u.yr * d).to(u.km/u.s, u.dimensionless_angles())
	vdec = (np.array(df.pmdec)*u.mas/u.yr * d).to(u.km/u.s, u.dimensionless_angles())
	v_t=np.sqrt(np.power(vra,2.)+np.power(vdec,2.)) # vtan
	# v_b as a proxy for v_z:
	c = coord.SkyCoord(ra=np.array(df.ra)*u.deg, dec=np.array(df.dec)*u.deg, distance=d,
	                  pm_ra_cosdec=np.array(df.pmra)*u.mas/u.yr,
	                  pm_dec=np.array(df.pmdec)*u.mas/u.yr)
	gal = c.galactic
	v_b = (gal.pm_b * gal.distance).to(u.km/u.s, u.dimensionless_angles()) # vb
	return v_t,v_b
	# print(vb)

# calculate absolute magnitude
def m_to_M(m, D):
    """
    Convert apparent magnitude to absolute magnitude.
    """
    return m - 5*np.log10(D)-10

#df["abs_G"] = m_to_M(df.phot_g_mean_mag.values, 1./df.parallax.values)
#df=df.dropna(subset=["abs_G"])


# calculates chisq
def calcChi(Prot,Prot_pre,Prot_err):
    # Prot: rotation periods
    # Prot_pre: predicted rotation periods
    # Prot_err: rotation period errors
    validv=0
    for i in range(len(Prot)):
        if Prot_err[i]==0 or Prot_err[i]==np.nan:
            Prot[i]=0
            Prot_pre[i]=0
            Prot_err[i]=1
            validv=validv+1
    avstedv=sum([(Prot[i]-Prot_pre[i])**2./Prot_err[i] for i in range(len(Prot_err))])/(len(Prot_pre)-validv)
    return avstedv
    
# calculates median relative error
def MRE(Prot,Prot_pre,Prot_err):
    # Prot: rotation periods
    # Prot_pre: predicted rotation periods
    # Prot_err: rotation period errors
    validv=0
    #print(Prot-Prot_pre)
    #print(Prot)
    meree=np.median([abs(Prot[i]-Prot_pre[i])/Prot[i] for i in range(len(Prot_err))])
    return meree


def readfits(filename):
    with fits.open(filename) as data:
        return(pd.DataFrame(data[1].data))


def fitpoints(x,y,order=1):
    z = np.polyfit(x,y,order)
    p = np.poly1d(z)
    return p


from tqdm import trange
import math
def movingMed_time(x,y,x_window,delta_x_window,minn_points=5,std_calc=False):
    x, y = zip(*sorted(zip(x,y)))
    x, y = np.array(x), np.array(y)
    # medians output
    x_med=np.ones(len(x))*np.nan
    y_med=np.ones(len(y))*np.nan
    if std_calc:
        y_std = np.ones(len(x))*np.nan
    # define the boundaries of the windows
    if len(x)==0:
        return x_med,y_med
    window_min=float(min(x))
    window_max=float(window_min+delta_x_window)
    
    # max time
    maxtime=max(x)
    
    # break when time window hits the end
    while window_max<=maxtime+x_window:
        seldf=(x>=window_min) & (x<=window_max) # get points between the window
        
        if sum(seldf)<minn_points:
            x_med[seldf]=np.nan
            y_med[seldf]=np.nan
            

        else:
            x_med[seldf]=np.median(x[seldf]) # all values for these indices are subsituded with median time
            y_med[seldf]=np.median(y[seldf]) # all values for these indices are subsituded with median flux
            if std_calc:
                y_std[seldf]=1.5*np.median(abs(y[seldf]-np.median(y[seldf])))/np.sqrt(sum(seldf))
            
        # slide the window
        window_min=window_min+delta_x_window
        window_max=window_max+delta_x_window
    if std_calc:
        return x_med[y_med==y_med], y_med[y_med==y_med], y_std[y_med==y_med]
    else:
        return x_med[y_med==y_med], y_med[y_med==y_med]

def calcmidval(x_med,y_med,x_val):
    sort_x_med,sort_y_med=zip(*sorted(zip(x_med,y_med)))
    sort_x_med=np.asarray(sort_x_med)
    sort_y_med=np.asarray(sort_y_med)
    if x_val in sort_x_med:
        m=(sort_x_med==x_val)
        #print(sort_y_med[m])
        return sort_y_med[m][0]
    else:
        for i in range(len(sort_x_med)):
            if sort_x_med[i]>x_val:
                if i==0:
                    return sort_y_med[0]
                else:
                    p=fitpoints([sort_x_med[i-1],sort_x_med[i]],[sort_y_med[i-1],sort_y_med[i]])
                    return p(x_val)
            return(sort_y_med[-1])
        
def calc_measure_disp(df,name,method='movingmedian',p=0):
    trytime=100
    agedisp=[]
    
    x,xerr=df['Age'].values,df['Age_err'].values
    y,yerr=df[name].values,df[name+'_ERR'].values
    
    if method=='movingmedian':
        for i in trange(trytime):
            df['newage']=np.array([np.random.normal(x[k], xerr[k],1)[0] for k in range(len(x))])
            df['newabun']=np.array([np.random.normal(y[k], yerr[k],1)[0] for k in range(len(y))])
            df['newmed']=np.array([calcmidval(df['med_age'],df['med_abun'],i) for i in df['newage']])
        
            agedisp.append(np.mean((df['newabun']-df['newmed'])**2.))
            
    elif method=='linear':
        for i in trange(trytime):
            df['newage']=np.array([np.random.normal(x[k], xerr[k],1)[0] for k in range(len(x))])
            df['newabun']=np.array([np.random.normal(y[k], yerr[k],1)[0] for k in range(len(y))])
            df['newmed']=p(df['newage'])
            
            agedisp.append(np.mean((df['newabun']-df['newmed'])**2.))
            
            
    return np.std(np.array(agedisp))**2.



def makeagemap_each(R,z,ageval,lims,binnum):
    x1,x2,y1,y2 = lims[0], lims[1], lims[2], lims[3]
    
    R = np.array(R)
    z = np.array(z)
    ageval = np.array(ageval) 
    
    xval = R
    yval = z
    wval = ageval 
    
    x1m,x2m,y1m,y2m = lims[0], lims[1], lims[2], lims[3]
    
    hist1,x2,y2 = np.histogram2d(xval, yval, weights = wval, bins= binnum, range = ((x1m,x2m), (y1m,y2m)))
    hist1_norm,x3,y3 = np.histogram2d(xval, yval, bins = binnum, range = ((x1m,x2m), (y1m,y2m)))

    image = hist1/hist1_norm  
    
    masked_array = np.ma.array (image, mask=np.isnan(image))
    
    return masked_array

def importage(agecode,dist=True):
    """
    agecode can be "LAMOST", "GALAH", or "ALL"
    """
    if agecode=="LAMOST":
        if dist:
            return pd.read_pickle(home_dic+'/Desktop/CreateDataTable/cannonages/allLAMOST_cut_dist.pkl')
        if not dist:
            return pd.read_pickle(home_dic+'/Desktop/CreateDataTable/cannonages/allLAMOST_cut.pkl')
    elif agecode=="GALAH":
            return pd.read_pickle(home_dic+'/Desktop/CreateDataTable/cannonages/galahages_cut.pkl')
    elif agecode=="ALL":
        if not dist:
            return pd.read_pickle(home_dic+'/Desktop/CreateDataTable/cannonages/allages_cut.pkl')
        if dist:
            return pd.read_pickle(home_dic+'/Desktop/CreateDataTable/cannonages/allages_cut_dist.pkl')

def importcm(cm1, cm2, cm3=False, cm4=False):
    """
    cm1, cm2 can be "LAMOST", "APOGEE", "RAVE", "TESS", "Kepler_prot", "Kepler_nonprot", "Kepler_all",
    "APOGEE", "GALAH"
    """
    allfiles=glob.glob(home_dic+'/Desktop/CreateDataTable/Data/*_cm_*.pkl')
    cmnumb=[]
    for i in allfiles:
        allnumb=sum([i=='cm' for i in i.split('_')])
        cmnumb.append(allnumb+1)

    allcm=[cm1,cm2,cm3,cm4]
    allcm_sub=[]
    if 'Kepler_all' in allcm:
        keplerall=True
        for i in allcm:
            if i=='Kepler_all':
                continue
            elif type(i) is not bool:
                allcm_sub.append(i)  
        allcm_sub.append("Kepler_nonprot")
        allcm_sub.append("Kepler_prot")    
    else:
        keplerall=False
        for i in allcm:
            if type(i) is not bool:
                allcm_sub.append(i)      
    if keplerall:
        cmlen=len(allcm_sub)-1
    else:
        cmlen=len(allcm_sub)
        
    m=(np.array(cmnumb)==cmlen)
    allfiles=np.array(allfiles)[m]
    
    if keplerall:
        foundfile=[0,0]
        for i in allfiles:
            m=np.zeros(len(allcm_sub)-2)
            for j in range(len(allcm_sub[:-2])):
                m[j]=(allcm_sub[:-2][j] in i)
            if sum(m)==len(m):
                if 'Kepler_nonprot' in i:    
                    filekepler_nonprot=pd.read_pickle(i)
                    filenamenonprot=i
                    foundfile[0]=1
                elif 'Kepler_prot' in i:
                    filekepler_prot=pd.read_pickle(i)
                    filenameprot=i
                    foundfile[1]=1
        if sum(foundfile)==2:
            print('found both stars in Kepler w/ and w/o prot with cm'+str(allcm))
            print('reading:'+filenamenonprot+','+filenameprot)
            return pd.concat([filekepler_nonprot,filekepler_prot])
        elif foundfile[0]==1:
            print('found stars in Kepler only w/o prot with cm'+str(allcm))
            return filekepler_nonprot
        elif foundfile[1]==1:
            print('found stars in Kepler only w/ prot with cm'+str(allcm))
            return filekepler_prot
        else:
            print('No stars with cm bewteen '+str(allcm))
            return None
    else:
        foundfile=0  
        for i in allfiles:
            m=np.zeros(len(allcm_sub))
            for j in range(len(allcm_sub)):
                m[j]=(allcm_sub[j] in i)
                if sum(m)==len(m):    
                    foundfile=1
                    print('found stars with cm '+str(allcm)+': '+i)
                    return pd.read_pickle(i)
        if foundfile==0:
            print('No stars with cm bewteen '+str(allcm))
            return None
            

def loaddata(datatype,agecode='ALL', dist=True, cm1='LAMOST', cm2='APOGEE', cm3=False, cm4=False):
    """
    datatype can be "age" (load age data), "cm"(load crossmatches), "LAMOSTdist" (load lamost dist and gaia cm),
     "LAMOSTkin" (load lamost kinematic data)
    """
    if datatype=="age":
        return importage(agecode, dist=dist)
    elif datatype=="cm":
        return importcm(cm1, cm2, cm3, cm4)
    elif datatype=="LAMOSTdist":
        return readfits(home_dic+'/Desktop/CreateDataTable/Data/LAMOST-dr5v3-gaia-edr3-coords-distances.fits')
    elif datatype=="LAMOSTkin":
        return readfits(home_dic+'/Desktop/CreateDataTable/Data/LAMOST-dr5v3-gaiaEDR3_orbit.fits')
    else:
        print("datatype can be 'age' (load age data), 'cm'(load crossmatches), 'LAMOSTdist' (load lamost dist and gaia cm), 'LAMOSTkin' (load lamost kinematic data)")

def convd2R(theta):
    return theta/180.*np.pi
    
def calcxyz(r_est,l,b):
    x = r_est * np.cos(convd2R(b)) * np.cos(convd2R(l)) / 1000.
    y = -r_est * np.sin(convd2R(l)) * np.cos(convd2R(b))/ 1000.
    z = r_est * np.sin(convd2R(b))/ 1000.
    r = np.sqrt(np.power(x - 8.2,2.) + np.power(y,2.))
    return x,y,z,r

import matplotlib
def getcolor(vrange,cm='plasma'):
    cmap = plt.cm.get_cmap(cm)
    norm = matplotlib.colors.Normalize(vmin=min(vrange), vmax=max(vrange))
    return [cmap(norm(i)) for i in vrange]

def crossmatch(df1,df2,sep=1.2):
    # df1
    c1 = SkyCoord([i*u.deg for i in df1['ra']], [i*u.deg for i in df1['dec']], frame='icrs')
    # apogee
    c2 = SkyCoord([i*u.deg for i in df2['ra']], [i*u.deg for i in df2['dec']], frame='icrs')
    # cross-match
    idx_sdss, d2d_sdss, d3d_sdss = c2.match_to_catalog_sky(c1)
    idx=idx_sdss[d2d_sdss<sep*u.arcsec]
    d2d=d2d_sdss[d2d_sdss<sep*u.arcsec]

    df1df2=df2[d2d_sdss<sep*u.arcsec]
    for i in df1.columns:
        df1df2[i]=df1.iloc[idx][i].values
        
    return df1df2

from scipy.interpolate import interp1d
def getRb(age, feh, fehoffset=0):
    feh0 = pd.read_csv(home_dic+'/Desktop/FeHGrad/centralfeh.csv') 
    slopefeh = pd.read_csv(home_dic+'/Desktop/FeHGrad/slopefeh.csv')
    
    func_m = interp1d(slopefeh['age'], slopefeh['slope'],'linear')
    func_b = interp1d(feh0['age'], feh0['feh0']+fehoffset, 'linear')
    
    Rb = np.zeros(len(age))
    for i in trange(len(age)):
        try:
            m = func_m(age[i])
            b = func_b(age[i])
            Rb[i] = (feh[i]-b)/m
        except:
            Rb[i] = np.nan
    return Rb

def bprp_to_teff(bprp):
    """
    Calculate photometric Teff from Gaia color (use dereddened color!)
    Args:
        bprp (array): Gaia G_BP colour minus Gaia G_RP colour.
    Returns:
        teffs (array): Photometric effective temperatures.
    """

    coeffs = [8959.8112335205078, -4801.5566310882568, 1931.4756631851196,
            -2445.9980716705322, 2669.0248055458069, -1324.0671020746231,
            301.13205924630165, -25.923997443169355]
    """
    # Jason's updated parameters:
    coeffs = [-416.585, 39780.0, -84190.5, 85203.9, -48225.9, 15598.5,
              -2694.76, 192.865]
    """

    return np.polyval(coeffs[::-1], bprp)




def LouisTurnoverTime(Teff):
    if (Teff<=3480.):
        tauLouis = 10.**(6.52112823e-7*Teff**2. - 4.00355099e-3*Teff + 8.68234621)
    else:
        tauLouis = 10.**(-2.51904051e-10*Teff**3. + 3.73613409e-6*Teff**2. - 1.85566042e-2*Teff + 32.5950535)
    #renormalise to CS11 scale
    tau = tauLouis*13.79/35.54
 
    return tau

def tauc(Teffs):
    taus = np.zeros(len(Teffs))
    for i in range(len(taus)):
        Teff = Teffs[i]
        if (Teff<=3480.):
            tauLouis = 10.**(6.52112823e-7*Teff**2. - 4.00355099e-3*Teff + 8.68234621)
        else:
            tauLouis = 10.**(-2.51904051e-10*Teff**3. + 3.73613409e-6*Teff**2. - 1.85566042e-2*Teff + 32.5950535)
        #renormalise to CS11 scale
        taus[i] = tauLouis*13.79/35.54
 
    return taus


def vxvyvz_to_vphivrvz(x,y,z,vx,vy,vz):
    R,phi,Z=(np.sqrt(x**2.+y**2.),np.arctan2(x,y),z)
    vr= vx*np.cos(phi)+vy*np.sin(phi)
    vt= -vx*np.sin(phi)+vy*np.cos(phi)
    return vr, vt, vz


def Table_to_pandas(fn, input_id=1):
    data = fits.open(fn)
    boss_mwm = QTable(data[input_id].data)
    cols = []
    cols_drop = []
    for i in boss_mwm.columns:
        #print(boss_mwm[i][0])
        if np.size(boss_mwm[i][0])==1:
            cols.append(i)
        else:
            cols_drop.append(i)
    print(cols_drop)
    return boss_mwm[cols].to_pandas()



import jax
import numpy as np
import jax.numpy as jnp
from tinygp import kernels


import jaxopt
from tinygp import GaussianProcess, kernels, transforms
from functools import partial
import arviz as az
import corner


def mean_function_both(params, X):
    # Prot broken low
    teffnorm = (7000.-X[0])/(7000.-params["teff_cut"])
    prot = jnp.power(10.,X[1])
    # Prot broken low
    stepfunc_low_prot = 1.0 / (1.0 + jnp.exp(-(jnp.log10(params['prot_cut']) - X[1]) / abs(params['w_prot'])))
    stepfunc_high_prot = 1.0 / (1.0 + jnp.exp(-(-jnp.log10(params['prot_cut']) + X[1]) / abs(params['w_prot'])))
    
    mod_high_prot = jnp.power(prot,params["b"])
    mod_high_prot = mod_high_prot*stepfunc_high_prot
    
    mod_low_prot = jnp.power(prot,params["b2"])*jnp.power(params["prot_cut"],params["b"]-params["b2"])
    mod_low_prot = mod_low_prot*stepfunc_low_prot
    
    prot_func = mod_high_prot+mod_low_prot
    
    # teff broken low
    stepfunc_high_teff = 1.0 / (1.0 + jnp.exp(-(1. - teffnorm) / abs(params['w_teff'])))
    stepfunc_low_teff = 1.0 / (1.0 + jnp.exp(-(-1. + teffnorm) / abs(params['w_teff'])))
    
    
    mod_high_teff = jnp.power(teffnorm-params["c"],params["d"])
    mod_high_teff = mod_high_teff*stepfunc_high_teff
    
    mod_low_teff = jnp.power(teffnorm-params["c"],params["d2"])*jnp.power(1.-params["c"],params["d"]-params["d2"])
    mod_low_teff = mod_low_teff*stepfunc_low_teff
    teff_func = mod_high_teff+mod_low_teff
    
    return params["a"]*prot_func*teff_func


def build_gp(params, X, yerr):
    kernel = jnp.exp(params["log_amp"]) * transforms.Linear(
        jnp.array([jnp.exp(-params["log_scale1"]), jnp.exp(-params["log_scale2"])]),
        kernels.ExpSquared()
    )

    return GaussianProcess(
        kernel, X, diag=yerr**2,mean=partial(mean_function_both, params)
    )



def mean_function_both_FC(params, X):
    # Prot broken low
    teffnorm = (3500.-X[0])/500.
    prot = jnp.power(10.,X[1])
    # Prot broken low
    stepfunc_low_prot = 1.0 / (1.0 + jnp.exp(-(jnp.log10(params['prot_cut']) - X[1]) / abs(params['w_prot'])))
    stepfunc_high_prot = 1.0 / (1.0 + jnp.exp(-(-jnp.log10(params['prot_cut']) + X[1]) / abs(params['w_prot'])))
    
    mod_high_prot = jnp.power(prot,params["b"])
    mod_high_prot = mod_high_prot*stepfunc_high_prot
    
    mod_low_prot = jnp.power(prot,params["b2"])*jnp.power(params["prot_cut"],params["b"]-params["b2"])
    mod_low_prot = mod_low_prot*stepfunc_low_prot
    
    prot_func = mod_high_prot+mod_low_prot
    
    # teff broken low
    mod_high_teff = jnp.power(teffnorm-params["c"],params["d"])
    teff_func = mod_high_teff
    
    return params["a"]*prot_func*teff_func


def build_gp_FC(params, X, yerr):
    kernel = jnp.exp(params["log_amp"]) * transforms.Linear(
        jnp.array([jnp.exp(-params["log_scale1"]), jnp.exp(-params["log_scale2"])]),
        kernels.ExpSquared()
    )

    return GaussianProcess(
        kernel, X, diag=yerr**2,mean=partial(mean_function_both_FC, params)
    )

allkeys = ['a', 'b', 'b2', 'c', 'd', 'd2', 'log_amp', 'log_scale1',
           'log_scale2', 'prot_cut', 'teff_cut', 'w_prot', 'w_teff']
sample_PC = np.load(home_dic+'/Desktop/NewGyroKineage/sample_PC.npy')

allkeys_fc = ['a', 'b', 'b2', 'c', 'd', 'log_amp', 'log_scale1',
           'log_scale2', 'prot_cut', 'w_prot']
sample_FC = np.load(home_dic+'/Desktop/NewGyroKineage/sample_FC.npy')

sample_pc_fig = np.zeros(np.shape(sample_FC))
sample_pc_fig[0:5,:] = sample_PC[0:5,:]
sample_pc_fig[5:9,:] = sample_PC[6:10,:]
sample_pc_fig[9,:] = sample_PC[11,:]


m = ((sample_FC[6,:]>5)&(sample_FC[6,:]<10))
sample_FC = sample_FC[:,m]
sample_PC = pd.DataFrame(sample_PC.T, columns=allkeys)
sample_PC = sample_PC.sample(n=100)

def GP_gyro_PC(X, sample):
    X_t = np.load(home_dic+'/Desktop/NewGyroKineage/X.npy')
    y_t = np.load(home_dic+'/Desktop/NewGyroKineage/yerr.npy')
    y = np.load(home_dic+'/Desktop/NewGyroKineage/y.npy')
    allkeys = ['a', 'b', 'b2', 'c', 'd', 'd2', 'log_amp', 'log_scale1',
           'log_scale2', 'prot_cut', 'teff_cut', 'w_prot', 'w_teff']
    outputage = np.zeros((len(sample),len(X)))
    for i in trange(len(sample)):
        val = sample.iloc[i][allkeys]
        uncorr_gp = build_gp(val, X_t, y_t)
        outputage[i,:] = uncorr_gp.condition(y, X).gp.loc.reshape(len(X))
    #print(outputage)
    ages = np.zeros(len(X))
    ages_p = np.zeros(len(X))
    ages_m = np.zeros(len(X))
    for i in range(len(X)):
        mcmc = np.percentile((outputage)[:, i][(outputage)[:, i]==(outputage)[:, i]], [16, 50, 84])
        q = np.diff(mcmc)
        ages[i] = mcmc[1]
        ages_m[i] = -q[0]
        ages_p[i] = q[1]
    return ages, ages_m, ages_p


sample_FC = pd.DataFrame(sample_FC.T, columns=allkeys_fc)
sample_FC = sample_FC.sample(n=100)

def GP_gyro_FC(X, sample):
    X_t = np.load(home_dic+'/Desktop/NewGyroKineage/X_fc.npy')
    y_t = np.load(home_dic+'/Desktop/NewGyroKineage/yerr_fc.npy')
    y = np.load(home_dic+'/Desktop/NewGyroKineage/y_fc.npy')
    allkeys = ['a', 'b', 'b2', 'c', 'd', 'log_amp', 'log_scale1',
           'log_scale2', 'prot_cut', 'w_prot']
    outputage = np.zeros((len(sample),len(X)))
    for i in trange(len(sample)):
        val = sample.iloc[i][allkeys]
        uncorr_gp = build_gp_FC(val, X_t, y_t)
        outputage[i,:] = uncorr_gp.condition(y, X).gp.loc.reshape(len(X))
        #print(outputage[i,:])
    ages = np.zeros(len(X))
    ages_p = np.zeros(len(X))
    ages_m = np.zeros(len(X))
    for i in range(len(X)):
        try:
            mcmc = np.percentile((outputage)[:, i][(outputage)[:, i]==(outputage)[:, i]], [16, 50, 84])
            q = np.diff(mcmc)
            ages[i] = mcmc[1]
            ages_m[i] = -q[0]
            ages_p[i] = q[1]
        except:
            ages[i], ages_m[i], ages_p[i] = np.nan, np.nan, np.nan
    return ages, ages_m, ages_p


def GP_gyro(X, MG):
    jaogap = fitpoints([3560/2+3526.5/2, 3427.36/2+3395.1/2], [10.09, 10.24])
    m_pc = (MG<jaogap(X[:,0]))
    ages_all = np.zeros(len(X))
    ages_p_all = np.zeros(len(X))
    ages_m_all = np.zeros(len(X))
    
    ages_all[m_pc], ages_p_all[m_pc], ages_m_all[m_pc] = GP_gyro_PC(X[m_pc,:], sample_PC)
    ages_all[~m_pc], ages_p_all[~m_pc], ages_m_all[~m_pc] = GP_gyro_FC(X[~m_pc,:], sample_FC)
    return ages_all, ages_m_all, ages_p_all

def loadkinematic_dr3():
    return pd.read_pickle(home_dic+'/Desktop/NewGyroKineage/dr3_kinematic.pkl')

from astropy.io import ascii
def loadkepler_prot():
    kepler = ascii.read(home_dic+'/Desktop/NewGyroKineage/santos2021.txt').to_pandas()
    gaiakepler = readfits(home_dic+'/Desktop/AgeBinary/kepler_dr3_1arcsec.fits')
    mcquillan = ascii.read(home_dic+'/Desktop/FirstYearML/mcquillan2014.txt').to_pandas()
    mcquillan['Prot'] = mcquillan['PRot']
    mcquillan['Prot_err'] = mcquillan['e_PRot']
    kepler['Prot_err'] = kepler['E_Prot']
    kepler = pd.concat([kepler[['KIC', 'Prot', 'Prot_err','Sph']], mcquillan[['KIC', 'Prot', 'Prot_err','Rper']]])
    kepler = kepler.drop_duplicates('KIC', keep='first')
    kepler = pd.merge(kepler, gaiakepler, left_on='KIC',
                 right_on='kepid',how='inner')
    return kepler 

def downloadAPOGEE_DR17(data, filename='./wget_apogeedr17_spectra.txt'):
    master1 = 'wget -P spectra-reference-aspcapStar/ -np -xnH --cut-dirs 9 --no-check-certificate --user sdss --password 2.5-meters -r https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec/'
    data = data.reset_index(drop=True)
    telescope = data['TELESCOPE']
    field = data['FIELD']
    file = data['FILE']
    apoid = data['APOGEE_ID']

    paths = []
    for indx, i in enumerate(file):
        paths.append(master1+telescope[indx]+str('/')+field[indx]+str('/aspcapStar-dr17-')+apoid[indx]+str('.fits')) 
        print(master1+telescope[indx]+str('/')+field[indx]+str('/aspcapStar-dr17-')+apoid[indx]+str('.fits'))
    np.savetxt(filename, paths, fmt = "%s")
    return 1

def loadWB():
    # Binary: https://zenodo.org/record/4435257#.Yr33Ti1h3BI
    binary = readfits(home_dic+'/Desktop/AgeBinary/all_columns_catalog.fits')
    return binary

def pf(lctime, lcmags, P, medlctime = 0):
    if medlctime==0:
        t_fold=(lctime-np.median(lctime))-np.round((lctime-np.median(lctime))/P)*P
    else:
        t_fold=(lctime-medlctime)-np.round((lctime-medlctime)/P)*P
    return t_fold, lcmags

def gmag_to_Vmag(gmag, bprp):
    return 0.02704-0.01424*bprp+0.2156*bprp**2-0.01426*bprp**3+gmag

def loadmhxgboost():
    return pd.read_csv(home_dic+'/Desktop/StarAgeComp/table-1.csv')


def addkinematic(df, id_name = 'source_id'):
    data = fits.open(home_dic+'/Desktop/NewGyroKineage/dr3-rv-good-plx-MilkyWayPotential2022-joined.fits')
    kinematic = data[1].data
    print('Finished reading in data!')
    ids = pd.DataFrame(kinematic.source_id, 
                  columns=['source_id'])
    checklist = ids['source_id'].isin(df[id_name])
    print('Finished matching source id, total %d stars'%(sum(checklist)))
    kinematic_dr3 = kinematic[checklist]
    list(kinematic_dr3.columns)
    
    
    kinematic_dr3 = pd.DataFrame(np.array((kinematic_dr3.source_id,
                                          kinematic_dr3.xyz[:,0],
                                          kinematic_dr3.xyz[:,1],
                                          kinematic_dr3.xyz[:,2],
                                          kinematic_dr3.vxyz[:,0],
                                          kinematic_dr3.vxyz[:,1],
                                          kinematic_dr3.vxyz[:,2],
                                          kinematic_dr3.actions[:,0],
                                          kinematic_dr3.actions[:,1],
                                           kinematic_dr3.actions[:,2],
                                          kinematic_dr3.E,
                                          kinematic_dr3.L[:,0],
                                          kinematic_dr3.L[:,1],
                                          kinematic_dr3.L[:,2],
                                          kinematic_dr3.ecc,
                                          kinematic_dr3.parallax,
                                          kinematic_dr3.ra,
                                          kinematic_dr3.dec,
                                          kinematic_dr3.phot_g_mean_mag,
                                          kinematic_dr3.phot_bp_mean_mag,
                                          kinematic_dr3.phot_rp_mean_mag,
                                          kinematic_dr3.ruwe,
					  kinematic_dr3.z_max,
					  kinematic_dr3.r_apo/2+kinematic_dr3.r_per/2),dtype=str).T,
                                columns=['source_id','x','y','z',
                                        'vx','vy','vz','Jx','Jy',
                                        'Jz','E','Lx','Ly','Lz','e',
                                         'parallax','ra','dec','phot_g_mean_mag',
                                        'phot_bp_mean_mag','phot_rp_mean_mag',
                                        'ruwe','z_max','Rg'])
    
    for i in kinematic_dr3.columns:
        if i=='source_id':
            kinematic_dr3[i] = [int(j) for j in kinematic_dr3[i]]
            continue
        kinematic_dr3[i] = [float(j) for j in kinematic_dr3[i]]
        
    df = pd.merge(df, kinematic_dr3,
             left_on=id_name,right_on='source_id',
             how='left')
    
    return df

def g_to_v(g, bprp):
    return g+0.02704-0.01424*bprp+0.2156*bprp**2-0.01426*bprp**3

def load_mwm_ipl3():
    filename = home_dic+'/Desktop/APOGEE_IPL/APOGEE_IPL3/astraAllStarASPCAP-0.5.0.fits'
    # This function is my way to get ride of columns that are more than 1D, since I don't think pandas supports that 
    def Table_to_pandas(fn):
        data = fits.open(fn)
        df = QTable(data[2].data)
        cols = []
        cols_drop = []
        for i in df.columns:
            #print(boss_mwm[i][0])
            if np.size(df[i][0])==1:
                cols.append(i)
            else:
                cols_drop.append(i)
        print(cols_drop)
        return df[cols].to_pandas()
    return Table_to_pandas(filename)

def dem_reg(x_obs, y_obs, x_obs_err, y_obs_err):
    mean_x = x_obs[0]
    x_obs = x_obs-mean_x

    mean_y = y_obs[0]
    y_obs = y_obs-mean_y

    sigma_x = np.std(x_obs_err)
    sigma_y = np.std(y_obs_err)

    # For centered data, compute the sums of squares:
    Sxx = np.sum(x_obs**2)
    Syy = np.sum(y_obs**2)
    Sxy = np.sum(x_obs * y_obs)

    lam =  (sigma_y**2) / (sigma_x**2)  # In this example, lam = 1
    beta_deming = (Syy - lam * Sxx + np.sqrt((Syy - lam * Sxx)**2 + 4 * lam * Sxy**2)) / (2 * Sxy)

    emp_m_dem = beta_deming
    emp_b_dem = mean_y-beta_deming*mean_x
    
    #(x_obs+mean_x), (x_obs*beta_deming+mean_y)
    return np.poly1d([emp_m_dem, emp_b_dem])

def load_apogee_dr17():
    return Table_to_pandas(home_dic+'/Desktop/Dwarf/allStar-dr17-synspec_rev1.fits')

def load_all_prot(kinematic=False, extinct=False):
    if kinematic==True and extinct==False:
        return pd.read_csv(home_dic+'/Desktop/KeplerAges/All_prot_kin.csv')
    if kinematic==True and extinct==True:
        return pd.read_csv(home_dic+'/Desktop/KeplerAges/All_prot_dered.csv')
    return pd.read_csv(home_dic+'/Desktop/KeplerAges/All_prot.csv')


def bprp_to_VG(bprp):
    color_inds = pd.read_csv('/Users/lu.3234/Desktop/JzKineAge/Gaia_color_swaps.csv')
    m_sel_VG = ((color_inds['Y']=='V-G ')&(color_inds['X']=='BP-RP'))
    color_inds_ind = color_inds[m_sel_VG].values[0]
    coeffs = color_inds_ind[6:16].astype(float)
    return np.polyval(coeffs[::-1], bprp)

def bprp_to_BG(bprp):
    color_inds = pd.read_csv('/Users/lu.3234/Desktop/JzKineAge/Gaia_color_swaps.csv')
    m_sel_BG = ((color_inds['Y']=='B-G ')&(color_inds['X']=='BP-RP'))
    m_sel_BG = m_sel_BG&(color_inds['Validity']=='Dwarfs')
    color_inds_ind = color_inds[m_sel_BG].values[0]
    coeffs = color_inds_ind[6:16].astype(float)
    return np.polyval(coeffs[::-1], bprp)

def bprp_to_BV(bprp):
    return bprp_to_BG(bprp)-bprp_to_VG(bprp)


def load_Hunt_cluster(cluster='all', Prob = 0.8):
    #cluster = pd.read_csv('Hunt_cluster.csv')
    members = pd.read_csv('Hunt_member.csv')
    if cluster == 'all':
        return members.loc[members['Prob']>0.8].reset_index(drop=True)
    else:
        return members.loc[(members['Prob']>0.8)&(members['Name']==cluster)].reset_index(drop=True)

def load_galahdr3():
    galah = readfits(home_dic+'/Desktop/FeHGrad/GALAH_DR3_main_allstar_v2.fits')
    galah_gaia = readfits(home_dic+'/Desktop/Jdot_haloM/GALAH_DR3_VAC_GaiaEDR3_v2.fits')
    return pd.merge(galah, galah_gaia, left_on='dr3_source_id', right_on='dr3_source_id', how='inner')

def correct_ztf(ztfprot, trueprot, freq_tol = 0.05):
    Pztf = ztfprot.copy()
    Ptess = trueprot.copy()

    m_nonan = ((Pztf==Pztf)&(Ptess==Ptess))
    
    fobs = 1.0 / Pztf
    ftrue = 1.0 / Ptess
    
    # Candidate frequencies
    fcands = [
        np.abs(1 - fobs),       # 1-f
        1 + fobs,               # 1+f
        np.abs(2 - fobs),       # 2-f
        2 + fobs,               # 2+f
    ]
    
    fcands = np.vstack(fcands)
    fcands[fcands <= 0] = np.inf
    
    # Best alias
    idx = np.full(len(Pztf), -1, dtype=int)
    idx[m_nonan] = np.nanargmin(np.abs(fcands[:, m_nonan] - ftrue[m_nonan]), axis=0)
    
    fcorr = np.full(len(Pztf), np.nan)
    m = idx >= 0
    fcorr[m] = fcands[idx[m], np.where(m)[0]]
    
    Palias = np.stack([
        1.0/np.abs(1.0/Ptess - 1),
        1.0/(1.0/Ptess + 1),
        1.0/np.abs(1.0/Ptess - 2),
        1.0/(1.0/Ptess + 2),
    ], axis=1)
    
    
    # Distance from observed frequency to nearest alias
    alias_residual = np.min(np.abs(np.log10(Pztf[:, None]) - np.log10(Palias)), axis=1)
    
    alias_mask = (alias_residual) < freq_tol

    #print(Pztf[alias_mask])
    Pztf_corr = 1.0 / fcorr
    
    m_1_to_1 = (abs(Pztf_corr-Ptess)<0.1*(Ptess))|(abs(Pztf_corr-Ptess*2)<0.1*(Ptess*2))|(abs(Pztf_corr-Ptess/2)<0.1*(Ptess/2))

    Pztf[alias_mask&m_1_to_1] = 1.0 / fcorr[alias_mask&m_1_to_1]

    #Pztf[alias_mask] = 1.0 / fcorr[alias_mask]
    
    return Pztf, alias_mask&m_1_to_1





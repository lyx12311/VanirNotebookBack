import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import astropy.utils as au
from astropy.io import fits
import astropy.coordinates as coord

from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
import glob

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
   'text.usetex': False,
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

# for plotting results for importance and predict vs true
def plot_result(actrualF,importance,prediction,y_test,y_test_err,topn=20):
    # inputs:
    # actrualF: feature used in training (output from my_randF_mask)
    # importance/prediction: output from my_randF function
    # topn: how many features to plot (default=20)
    # X: features, if X is inputed then plot feature vs Prot
    # y_test: tested values
    # y_test_err: tested values errors
    
    # output: 
    # my_xticks: importance of features in decending order
    
    topn=min([topn,len(actrualF)])
    # zip the importance with its feature name
    list1 = list(zip(actrualF,importance))
    # sort the zipped list
    decend=sorted(list1, key=lambda x:x[1],reverse=True)
    #print(decend)

    # how many features to plot 
    x=range(topn)
    
    ####################  get most important features ############################################################
    y_val=[decend[i][1] for i in range(topn)]
    my_xticks=[decend[i][0] for i in range(topn)]

    plt.figure(figsize=(20,5))
    plt.title('Most important features',fontsize=25)
    plt.xticks(x, my_xticks)
    plt.plot(x, y_val,'k-')
    plt.xlim([min(x),max(x)])
    plt.xticks(rotation=90)
    plt.ylabel('importance')
    ####################  get most important features ############################################################

    # prediction vs true
    plt.figure(figsize=(20,8))
    plt.subplot(1,2,1)
    plt.plot(sorted(prediction),sorted(prediction),'k-',label='y=x')
    plt.plot(sorted(prediction),sorted(1.1*prediction),'b--',label='10% Error')
    plt.plot(sorted(prediction),sorted(0.9*prediction),'b--')
    plt.plot(y_test,prediction,'r.',Markersize=3,alpha=0.2)
    plt.ylabel('Predicted Period')
    plt.xlabel('True Period')
    plt.ylim([0,max(prediction)])
    plt.xlim([0,max(prediction)])
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(sorted(prediction),sorted(prediction),'k-',label='y=x')
    plt.plot(sorted(prediction),sorted(1.1*prediction),'b--',label='10% Error')
    plt.plot(sorted(prediction),sorted(0.9*prediction),'b--')
    plt.errorbar(y_test,prediction,xerr=y_test_err,fmt='r.',Markersize=3,alpha=0.2)
    plt.ylabel('Predicted Period')
    plt.xlabel('True Period')
    plt.ylim([0,max(prediction)])
    plt.xlim([0,max(prediction)])
    plt.legend()
    #plt.savefig('RF.png')
    
    avstedv=MRE(y_test,prediction,y_test_err)
    print('Median relative error is: ',avstedv)
    return(my_xticks)



# plot different features vs Prot
def plot_corr(df,my_xticks,logplotarg=[],logarg=[]):
    # df: dataframe
    # my_xticks: features to plot against Prot
    # logplotarg: arguments to plot in loglog space
    # logarg: which log to plot
    
    # add in Prot
    Prot=df.Prot
    df=df[my_xticks].dropna()
    Prot=Prot[df.index]
    topn=len(my_xticks)
    # get subplot config
    com_mul=[] 
    # get all multiplier
    for i in range(1,topn):
        if float(topn)/float(i)-int(float(topn)/float(i))==0:
            com_mul.append(i)
        
    # total rows and columns
    col=int(np.median(com_mul))
    row=int(topn/col)
    if col*row<topn:
        if col<row:
            row=row+1
        else:
            col=col+1
        
    # plot feature vs Prot
    plt.figure(figsize=(int(topn*2.5),int(topn*2.5)))
    for i in range(topn):
        plt.subplot(row,col,i+1)
        featurep=df[my_xticks[i]]
        if my_xticks[i] in logplotarg:
            if logarg=='loglog':
                plt.loglog(Prot,featurep,'k.',markersize=1)
            elif logarg=='logx':
                plt.semilogx(Prot,featurep,'k.',markersize=1)
            elif logarg=='logy':
                plt.semilogy(Prot,featurep,'k.',markersize=1)
            else:
                raise SyntaxError("Log scale input not recognized!")
        else:
            plt.plot(Prot,featurep,'k.',markersize=1)
        plt.title(my_xticks[i],fontsize=25)
        stddata=np.std(featurep)
        #print([np.median(featurep)-3*stddata,np.median(featurep)+3*stddata])
        plt.ylim([np.median(featurep)-3*stddata,np.median(featurep)+3*stddata])
        plt.xlabel('Prot')
        plt.ylabel(my_xticks[i])
        #plt.tight_layout()


############################# RF training #########################################
# use only a couple of features 
def my_randF_SL(df,traind,testF,X_train_ind=[],X_test_ind=[],chisq_out=False,MREout=False,n_estimators=100, criterion='mse', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False):
    # df: dataframe to train with all the features including Prot and Prot_err
    # traind: fraction of data use to train
    # testF: training feature names
    # X_train_ind: KID for training stars
    # X_test_ind: KID for testing stars
    # chisq_out: output only median relative error?
   
    print('regr,importance,actrualF,KID_train,KID_test,predictp,avstedv,avMRE = my_randF_SL(df,traind,testF,chisq_out=0,MREout=False,hyperp=[])\n')

    if len(X_train_ind)==0:
        print('Fraction of data used to train:',traind)
    else:
        print('Training KID specified!\n')
        print('Estimated fraction of data used to train:',len(X_train_ind)/len(df['Prot']))
    print('# Features used to train:',len(testF))
    print('Features used to train:',testF)

    fl=len(df.columns) # how many features
    keys=range(fl)
    flib=dict(zip(keys, df.columns))
    
    featl_o=len(df.Prot) # old feature length before dropping
    
    actrualF=[] # actrual feature used
    # fill in feature array
    lenX=0
    missingf=[]
    for i in df.columns:
        feature=df[i].values
        if (type(feature[0]) is not str) and (i in testF):
            if sum(np.isnan(feature))<0.1*featl_o:
                lenX=lenX+1
                actrualF.append(i)
            else:
                missingf.append(i)
            
    X=df[actrualF]
    X=X.replace([np.inf, -np.inf], np.nan)
    X=X.dropna()

    featl=np.shape(X)[0]
    #print(featl)
    print(str(featl_o)+' stars in dataframe!')
    if len(missingf)!=0:
        print('Missing features:',missingf)
    if (featl_o-featl)!=0:
        print('Missing '+ str(featl_o-featl)+' stars from null values in data!\n')

    print(str(featl)+' total stars used for RF!')
    

    #print(X_train_ind)

    if len(X_train_ind)==0:
        # output
        y=df.Prot[X.index].values
        y_err=df.Prot_err[X.index].values
        KID_ar=df.KID[X.index].values
        X=X.values
	
        Ntrain = int(traind*featl)
        # Choose stars at random and split.
        shuffle_inds = np.arange(len(y))
        np.random.shuffle(shuffle_inds)
        train_inds = shuffle_inds[:Ntrain]
        test_inds = shuffle_inds[Ntrain:]
	
        y_train, y_train_err, KID_train, X_train = y[train_inds], y_err[train_inds],KID_ar[train_inds],X[train_inds, :]
        y_test, y_test_err, KID_test, X_test = y[test_inds], y_err[test_inds],KID_ar[test_inds],X[test_inds, :]
	
        test_inds,y_test, y_test_err, KID_test, X_test=zip(*sorted(zip(test_inds,y_test, y_test_err, KID_test, X_test)))
        test_inds=np.array(test_inds)
        y_test=np.array(y_test)
        y_test_err=np.array(y_test_err)
        KID_test=np.array(KID_test)
        X_test=np.asarray(X_test)
	
    else:
        datafT=df.loc[X.index].loc[df['KID'].isin(X_train_ind)]
        datafTes=df.loc[X.index].loc[df['KID'].isin(X_test_ind)]
        y_train, y_train_err,X_train = datafT.Prot.values, datafT.Prot_err.values,X.loc[df['KID'].isin(X_train_ind)].values
        y_test, y_test_err,X_test = datafTes.Prot.values, datafTes.Prot_err.values,X.loc[df['KID'].isin(X_test_ind)].values
    print(str(len(y_train))+' training stars!')



    # run random forest
    regr = RandomForestRegressor(n_estimators=n_estimators, criterion=criterion, max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, min_weight_fraction_leaf=min_weight_fraction_leaf, max_features=max_features, max_leaf_nodes=max_leaf_nodes, min_impurity_decrease=min_impurity_decrease, min_impurity_split=min_impurity_split, bootstrap=bootstrap, oob_score=oob_score, n_jobs=n_jobs, random_state=random_state, verbose=verbose, warm_start=warm_start)
    regr.fit(X_train, y_train)  
    
    

    # get the importance of each feature
    importance=regr.feature_importances_
    
    print('Finished training! Making predictions!')
    # make prediction
    predictp=regr.predict(X_test)
    print('Finished predicting! Calculating chi^2!')
     
    # calculate chisq and MRE
    avMRE=MRE(y_test,predictp,y_test_err)
    avstedv=calcChi(y_test,predictp,y_test_err)

    print('Median Relative Error is:',avMRE)
    print('Average Chi^2 is:',avstedv)
    
    if chisq_out:
        if MREout:
            print('Finished!')
            return avstedv,avMRE
        else:
            print('Finished!')
            return avstedv
    elif MREout:
        print('Finished!')
        return avMRE
    else:
        if len(X_train_ind)!=0:
            KID_train=datafT.KID.values
            KID_test=datafTes.KID.values
            KID_train=[int(i) for i in KID_train]
            KID_test=[int(i) for i in KID_test]
        print('Finished!')
        return regr,importance,actrualF,KID_train,KID_test,predictp,avstedv,avMRE,X_test,y_test,X_train,y_train

def readfits(filename):
    with fits.open(filename) as data:
        return(pd.DataFrame(data[1].data,dtype='float64'))


def fitpoints(x,y,order=1):
    z = np.polyfit(x,y,order)
    p = np.poly1d(z)
    return p


from tqdm import trange
import math
def movingMed_time(x,y,x_window,delta_x_window):
    # medians output
    x_med=np.zeros(len(x))
    y_med=np.zeros(len(y))

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
        
        x_med[seldf]=np.median(x[seldf]) # all values for these indices are subsituded with median time
        y_med[seldf]=np.median(y[seldf]) # all values for these indices are subsituded with median flux
        
        # slide the window
        window_min=window_min+delta_x_window
        window_max=window_max+delta_x_window
    return x_med, y_med

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


def calc_disp(df,name,method='movingmedian',p=0):
    df_dropna=df.dropna(subset=['med_age',name,'med_abun',name+'_ERR'])

    if method=='movingmedian':
        # calcualte total dispersion
        totdisp=np.mean((df_dropna[name]-df_dropna['med_abun'])**2.)
        #print(totdisp)
    
        # calculate measurement dispersion
        meadisp=calc_measure_disp(df,name)
        
    elif method=='linear':
        # calcualte total dispersion
        totdisp=np.mean((df_dropna[name]-p(df_dropna['Age']))**2.)
        meadisp=calc_measure_disp(df,name,'linear',p)
        
    
    #print(df['Age_err'])
    # calculate int dispersion
    intdisp=np.sqrt(totdisp-meadisp)
    #print('total',totdisp,'measured',meadisp,'intr',intdisp)
    return np.sqrt(totdisp),intdisp


from scipy.ndimage import gaussian_filter
def plotTrends(df_abun_st_ha,df_abun_st_la,figname,method='movingmedian',order=1,xr=10):
    abun_apog=['M_H','C_FE','N_FE','O_FE','NA_FE','MG_FE','AL_FE','SI_FE','S_FE','K_FE','CA_FE','TI_FE',
           'V_FE','MN_FE','NI_FE','P_FE','CR_FE','CO_FE','S_FE']

    abun_apog_err=[i+'_ERR' for i in abun_apog]

    abund={'C_FE':'[C/','MG_FE':'[Mg/','AL_FE':'[Al/','SI_FE':'[Si/','S_FE':'[Sc/',
       'CA_FE':'[Ca/','TI_FE':'[Ti/','CR_FE':'[Cr/','MN_FE':'[Mn/','CO_FE':'[Co/',
       'NI_FE':'[Ni/','CU_FE':'[Cu/','O_FE':'[O/','NA_FE':'[Na/','K_FE':'[K/','V_FE':'[V/',
       'P_FE':'[P/','RB_FE':'[Rb/','N_FE':'[N/','S_FE':'[S/','M_H':'[Fe/'}
    
    abundlim={'C_FE':[-0.25,0.25],'MG_FE':[-0.25,0.25],'AL_FE':[-0.25,0.25],'SI_FE':[-0.25,0.25],
              'S_FE':[-0.25,0.25],'CA_FE':[-0.1,0.1],'TI_FE':[-0.25,0.25],'CR_FE':[-0.1,0.1],
              'MN_FE':[-0.25,0.25],'CO_FE':[-0.25,0.25],'NI_FE':[-0.1,0.1],'CU_FE':[-0.25,0.25],
              'O_FE':[-0.25,0.25],'NA_FE':[-0.25,0.25],'K_FE':[-0.25,0.25],'V_FE':[-0.25,0.25],
              'P_FE':[-0.25,0.25],'RB_FE':[-0.25,0.25],'N_FE':[-0.5,0.5],'M_H':[-0.1,0.1],}


    ## running median parameters
    agewindow=0.5
    slidwindow=0.1

    # **************************************** #

    Abunname=[]
    hi_dis=[]
    lo_dis=[]

    tot_hi_dis=[]
    tot_lo_dis=[]

    plt.figure(figsize=(22.5,25))
    for i in range(1,len(abun_apog)):
        clear_output(wait = True)
        
        df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha[abun_apog[i]]>-100]
        df_plot_ha=df_plot_ha.dropna(subset=['Age','Age_err',abun_apog[i],abun_apog[i]+"_ERR"])
        #df_plot_ha=df_plot_ha.loc[df_plot_ha['Age']<10]
        #df_plot_ha=df_plot_ha.loc[df_plot_ha['Age_err']<10]
        df_plot_ha=df_plot_ha.reset_index(drop=True)
    
        
        df_plot_la=df_abun_st_la.loc[df_abun_st_la[abun_apog[i]]>-100]
        df_plot_la=df_plot_la.dropna(subset=['Age','Age_err',abun_apog[i],abun_apog[i]+"_ERR"])
        #df_plot_la=df_plot_la.loc[df_plot_la['Age']<10]
        #df_plot_la=df_plot_la.loc[df_plot_la['Age_err']<10]
        df_plot_la=df_plot_la.reset_index(drop=True)
    
        #print('errer',df_plot_ha['Age_err'])
    
        df_plot_la=df_plot_la.sort_values(by=['Age'])
        df_plot_ha=df_plot_ha.sort_values(by=['Age'])
    
        #print(df_plot_ha['Age_err'])
    
        if method=='movingmedian':
            df_plot_la['med_age'],df_plot_la['med_abun']=movingMed_time(df_plot_la['Age'].values,
                                                                df_plot_la[abun_apog[i]].values,
                                                                agewindow,slidwindow)
            df_plot_ha['med_age'],df_plot_ha['med_abun']=movingMed_time(df_plot_ha['Age'].values,
                                                                df_plot_ha[abun_apog[i]].values,
                                                                agewindow,slidwindow)
    
            df_plot_la['med_abun']=gaussian_filter(df_plot_la['med_abun'], sigma=10)
            df_plot_ha['med_abun']=gaussian_filter(df_plot_ha['med_abun'], sigma=10)
    
            df_plot_la_new=df_plot_la.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
            df_plot_ha_new=df_plot_ha.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        
        elif method=='linear':
            if len(df_plot_la)==0 or len(df_plot_ha)==0:
                continue
            z = np.polyfit(df_plot_la['Age'].values,df_plot_la[abun_apog[i]].values,order)
            p_la = np.poly1d(z)
            z = np.polyfit(df_plot_ha['Age'].values,df_plot_ha[abun_apog[i]].values,order)
            p_ha = np.poly1d(z)
            
            
            df_plot_la['med_age'],df_plot_la['med_abun']=df_plot_la['Age'],p_la(df_plot_la['Age'])
            df_plot_ha['med_age'],df_plot_ha['med_abun']=df_plot_ha['Age'],p_ha(df_plot_ha['Age'])
    
            df_plot_la_new=df_plot_la.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
            df_plot_ha_new=df_plot_ha.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        
    
        plt.subplot(5,4,i+1)
        plt.errorbar(df_plot_la['Age'],df_plot_la[abun_apog[i]],fmt='b.',alpha=0.3,label='low-$\\alpha$')
        plt.errorbar(df_plot_ha['Age'],df_plot_ha[abun_apog[i]],fmt='r.',alpha=0.3,label='high-$\\alpha$')
    
        if len(df_plot_la['Age'])!=0:   
            print(abun_apog[i])
        
            plt.plot(df_plot_la['med_age'],df_plot_la['med_abun'],'b-',alpha=1,linewidth=2,label='low-$\\alpha$')
            plt.plot(df_plot_ha['med_age'],df_plot_ha['med_abun'],'r-',alpha=1,linewidth=2,label='high-$\\alpha$')
        
            #print(df_plot_ha['Age_err'])
            if method=='movingmedian':
                tot_la,disp_la=calc_disp(df_plot_la,abun_apog[i])
                tot_ha,disp_ha=calc_disp(df_plot_ha,abun_apog[i])
            elif method=='linear':
                tot_la,disp_la=calc_disp(df_plot_la,abun_apog[i],'linear',p_la)
                tot_ha,disp_ha=calc_disp(df_plot_ha,abun_apog[i],'linear',p_ha)
                
            #print(abun_apog[i])
            #plt.title('Dispersion: low-a:%.5f, high-a:%.5f,\n diffs:%.5f'%(disp_la,disp_ha,disp_ha-disp_la),fontsize=15)
    
    
            Abunname.append(abund[abun_apog[i]]+'Fe]')
            hi_dis.append(disp_ha)
            lo_dis.append(disp_la)
            tot_hi_dis.append(tot_ha)
            tot_lo_dis.append(tot_la)
            
            lawidth_la,lawidth_ha=tot_la,tot_ha
            minla,maxla,minha,maxha=df_plot_la['med_abun']-lawidth_la,df_plot_la['med_abun']+lawidth_la,df_plot_ha['med_abun']-lawidth_ha,df_plot_ha['med_abun']+lawidth_ha
            plt.fill_between(df_plot_la['med_age'],minla,maxla,color='b',alpha=0.2)
            plt.fill_between(df_plot_ha['med_age'],minha,maxha,color='r',alpha=0.2)
	    
        
        plt.ylabel(abund[abun_apog[i]]+'Fe]')
        plt.xlabel('Age [Gyr]')
        plt.ylim(abundlim[abun_apog[i]])
        plt.xlim([0,xr])
        plt.errorbar(xr-1,abundlim[abun_apog[i]][0]+0.2*abs(abundlim[abun_apog[i]][1]-abundlim[abun_apog[i]][0]),
                     yerr=np.median(df_plot_la[abun_apog[i]+'_ERR']),capsize=10,fmt='ko')
    
    
    
    #plt.legend()
    
    
    # For Fe
    df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha['M_H']>-100]
    #df_plot_ha=df_plot_ha.loc[df_plot_ha['Age_err']<10]
    df_plot_ha=df_plot_ha.reset_index(drop=True)
    
    df_plot_la=df_abun_st_la.loc[df_abun_st_la['M_H']>-100]
    #df_plot_la=df_plot_la.loc[df_plot_la['Age_err']<10]
    df_plot_la=df_plot_la.reset_index(drop=True)
    
    df_plot_la=df_plot_la.sort_values(by=['Age'])
    df_plot_ha=df_plot_ha.sort_values(by=['Age'])
    
    if method=='movingmedian':
        df_plot_la['med_age'],df_plot_la['med_abun']=movingMed_time(df_plot_la['Age'].values,
                                                                df_plot_la['M_H'].values,
                                                                agewindow,slidwindow)
        df_plot_ha['med_age'],df_plot_ha['med_abun']=movingMed_time(df_plot_ha['Age'].values,
                                                                df_plot_ha['M_H'].values,
                                                                agewindow,slidwindow)
    
        df_plot_la['med_abun']=gaussian_filter(df_plot_la['med_abun'], sigma=10)
        df_plot_ha['med_abun']=gaussian_filter(df_plot_ha['med_abun'], sigma=10)
    
        df_plot_la_new=df_plot_la.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        df_plot_ha_new=df_plot_ha.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        
    elif method=='linear':
        z = np.polyfit(df_plot_la['Age'].values,df_plot_la['M_H'].values,order)
        p_la = np.poly1d(z)
        z = np.polyfit(df_plot_ha['Age'].values,df_plot_ha['M_H'].values,order)
        p_ha = np.poly1d(z)
            
            
        df_plot_la['med_age'],df_plot_la['med_abun']=df_plot_la['Age'],p_la(df_plot_la['Age'])
        df_plot_ha['med_age'],df_plot_ha['med_abun']=df_plot_ha['Age'],p_ha(df_plot_ha['Age'])
    
        df_plot_la_new=df_plot_la.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        df_plot_ha_new=df_plot_ha.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
    
    plt.subplot(5,4,1)
    plt.errorbar(df_plot_la['Age'],df_plot_la['M_H'],fmt='b.',alpha=0.3,label='low-$\\alpha$')
    plt.errorbar(df_plot_ha['Age'],df_plot_ha['M_H'],fmt='r.',alpha=0.3,label='high-$\\alpha$')
    
    plt.plot(df_plot_la['med_age'],df_plot_la['med_abun'],'b-',alpha=1,linewidth=2,label='low-$\\alpha$')
    plt.plot(df_plot_ha['med_age'],df_plot_ha['med_abun'],'r-',alpha=1,linewidth=2,label='high-$\\alpha$')

    if method=='movingmedian':
            tot_la,disp_la=calc_disp(df_plot_la,'M_H')
            tot_ha,disp_ha=calc_disp(df_plot_ha,'M_H')
    elif method=='linear':
            tot_la,disp_la=calc_disp(df_plot_la,'M_H','linear',p_la)
            tot_ha,disp_ha=calc_disp(df_plot_ha,'M_H','linear',p_ha)
            
    #plt.title('Dispersion: low-a:%.5f, high-a:%.5f,\n diffs:%.5f'%(disp_la,disp_ha,disp_ha-disp_la),fontsize=15)

    Abunname.append('Fe')
    hi_dis.append(disp_ha)
    lo_dis.append(disp_la)
    tot_hi_dis.append(tot_ha)
    tot_lo_dis.append(tot_la)
    
    lawidth_la,lawidth_ha=tot_la,tot_ha
    minla,maxla,minha,maxha=df_plot_la['med_abun']-lawidth_la,df_plot_la['med_abun']+lawidth_la,df_plot_ha['med_abun']-lawidth_ha,df_plot_ha['med_abun']+lawidth_ha
    plt.fill_between(df_plot_la['med_age'],minla,maxla,color='b',alpha=0.2)
    plt.fill_between(df_plot_ha['med_age'],minha,maxha,color='r',alpha=0.2)

    plt.ylabel('[Fe/H]')
    plt.xlabel('Age [Gyr]')
    plt.ylim([-0.1,0.1])
    plt.xlim([0,xr])
    plt.errorbar(xr-1,-0.1+0.2*abs(0.2),yerr=np.median(df_plot_la['M_H_ERR']),capsize=10,fmt='ko')
		     
    #plt.legend()
 
    # for C/N
    df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha['C_FE']>-100]
    df_plot_ha=df_plot_ha.loc[df_plot_ha['N_FE']>-100]
    df_plot_ha['C_N']=df_plot_ha['C_FE']-df_plot_ha['N_FE']
    df_plot_ha['C_N_ERR']=np.sqrt(df_plot_ha['C_FE_ERR']**2.+df_plot_ha['N_FE_ERR']**2.)
    df_plot_ha=df_plot_ha.loc[df_plot_ha['Age_err']<10]
    df_plot_ha=df_plot_ha.reset_index(drop=True)
    
    df_plot_la=df_abun_st_la.loc[df_abun_st_la['C_FE']>-100]
    df_plot_la=df_plot_la.loc[df_plot_la['N_FE']>-100]
    df_plot_la['C_N']=df_plot_la['C_FE']-df_plot_la['N_FE']
    df_plot_la['C_N_ERR']=np.sqrt(df_plot_la['C_FE_ERR']**2.+df_plot_la['N_FE_ERR']**2.)
    df_plot_la=df_plot_la.loc[df_plot_la['Age_err']<10]
    df_plot_la=df_plot_la.reset_index(drop=True)

    df_plot_la=df_plot_la.sort_values(by=['Age'])
    df_plot_ha=df_plot_ha.sort_values(by=['Age'])    
    
    if method=='movingmedian':
        df_plot_la['med_age'],df_plot_la['med_abun']=movingMed_time(df_plot_la['Age'].values,
                                                                df_plot_la['C_N'].values,
                                                                agewindow,slidwindow)
        df_plot_ha['med_age'],df_plot_ha['med_abun']=movingMed_time(df_plot_ha['Age'].values,
                                                                df_plot_ha['C_N'].values,
                                                                agewindow,slidwindow)
    
        df_plot_la['med_abun']=gaussian_filter(df_plot_la['med_abun'], sigma=10)
        df_plot_ha['med_abun']=gaussian_filter(df_plot_ha['med_abun'], sigma=10)
    
        df_plot_la_new=df_plot_la.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        df_plot_ha_new=df_plot_ha.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        
    elif method=='linear':
        z = np.polyfit(df_plot_la['Age'].values,df_plot_la['C_N'].values,order)
        p_la = np.poly1d(z)
        z = np.polyfit(df_plot_ha['Age'].values,df_plot_ha['C_N'].values,order)
        p_ha = np.poly1d(z)
            
            
        df_plot_la['med_age'],df_plot_la['med_abun']=df_plot_la['Age'],p_la(df_plot_la['Age'])
        df_plot_ha['med_age'],df_plot_ha['med_abun']=df_plot_ha['Age'],p_ha(df_plot_ha['Age'])
    
        df_plot_la_new=df_plot_la.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        df_plot_ha_new=df_plot_ha.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        
    
    plt.subplot(5,4,20)
    plt.errorbar(df_plot_la['Age'],df_plot_la['C_N'],fmt='b.',alpha=0.3,label='low-$\\alpha$')
    plt.errorbar(df_plot_ha['Age'],df_plot_ha['C_N'],fmt='r.',alpha=0.3,label='high-$\\alpha$')
    
    plt.plot(df_plot_la['med_age'],df_plot_la['med_abun'],'b-',alpha=1,linewidth=2,label='low-$\\alpha$')
    plt.plot(df_plot_ha['med_age'],df_plot_ha['med_abun'],'r-',alpha=1,linewidth=2,label='high-$\\alpha$')

    if method=='movingmedian':
            tot_la,disp_la=calc_disp(df_plot_la,'C_N')
            tot_ha,disp_ha=calc_disp(df_plot_ha,'C_N')
    elif method=='linear':
            tot_la,disp_la=calc_disp(df_plot_la,'C_N','linear',p_la)
            tot_ha,disp_ha=calc_disp(df_plot_ha,'C_N','linear',p_ha)
  
    Abunname.append('[C/N]')
    hi_dis.append(disp_ha)
    lo_dis.append(disp_la)
    tot_hi_dis.append(tot_ha)
    tot_lo_dis.append(tot_la)

    #plt.title('Dispersion: low-a:%.5f, high-a:%.5f,\n diffs:%.5f'%(disp_la,disp_ha,disp_ha-disp_la),fontsize=15)

    lawidth_la,lawidth_ha=tot_la,tot_ha
    minla,maxla,minha,maxha=df_plot_la['med_abun']-lawidth_la,df_plot_la['med_abun']+lawidth_la,df_plot_ha['med_abun']-lawidth_ha,df_plot_ha['med_abun']+lawidth_ha
    plt.fill_between(df_plot_la['med_age'],minla,maxla,color='b',alpha=0.2)
    plt.fill_between(df_plot_ha['med_age'],minha,maxha,color='r',alpha=0.2)
    
    plt.ylabel('[C/N]')
    plt.xlabel('Age [Gyr]')
    plt.ylim([-0.5,0.5])
    plt.xlim([0,xr])
    plt.errorbar(xr-1,-0.5+0.2*abs(1),yerr=np.median(df_plot_la['C_N_ERR']),capsize=10,fmt='ko')
    
    #plt.legend()

    plt.tight_layout()

    plt.savefig(figname+'.png')
    return hi_dis, lo_dis, tot_hi_dis, tot_lo_dis, Abunname

def tempcheck(df_abun_st_ha,df_abun_st_la):
    abun_apog=['M_H','C_FE','N_FE','O_FE','NA_FE','MG_FE','AL_FE','SI_FE','S_FE','K_FE','CA_FE','TI_FE',
           'V_FE','MN_FE','NI_FE','P_FE','CR_FE','CO_FE','RB_FE']

    abun_apog_err=[i+'_ERR' for i in abun_apog]

    abund={'C_FE':'[CI/','MG_FE':'[MgI/','AL_FE':'[AlI/','SI_FE':'[SiI/','S_FE':'[ScI/',
       'CA_FE':'[CaI/','TI_FE':'[TiI/','CR_FE':'[CrI/','MN_FE':'[MnI/','CO_FE':'[CoI/',
       'NI_FE':'[NiI/','CU_FE':'[CuI/','O_FE':'[O/','NA_FE':'[Na/','K_FE':'[K/','V_FE':'[V/',
      'P_FE':'[P/','RB_FE':'[Rb/','N_FE':'[N/','M_H':'[Fe/'}

    plt.figure(figsize=(22.5,25))
    cm = plt.cm.get_cmap('viridis',5)
    for i in range(1,len(abun_apog)):
        df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha[abun_apog[i]]>-100]
        df_plot_ha=df_plot_ha.reset_index(drop=True)
    
        df_plot_la=df_abun_st_la.loc[df_abun_st_la[abun_apog[i]]>-100]
        df_plot_la=df_plot_la.reset_index(drop=True)
    
    
        plt.subplot(5,4,i+1)
        plt.scatter(df_plot_la['Age'],df_plot_la[abun_apog[i]],c=df_plot_la['TEFF'],vmin=tempcut-rangeTeff,vmax=tempcut+rangeTeff)
        plt.scatter(df_plot_ha['Age'],df_plot_ha[abun_apog[i]],c=df_plot_ha['TEFF'],vmin=tempcut-rangeTeff,vmax=tempcut+rangeTeff)
        plt.ylabel(abund[abun_apog[i]]+'Fe]')
        plt.xlabel('Age [Gyr]')
    
    # For Fe
    df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha['M_H']>-100]
    df_plot_ha=df_plot_ha.reset_index(drop=True)
    
    df_plot_la=df_abun_st_la.loc[df_abun_st_la['M_H']>-100]
    df_plot_la=df_plot_la.reset_index(drop=True)
    
    plt.subplot(5,4,1)
    plt.scatter(df_plot_la['Age'],df_plot_la['M_H'],c=df_plot_la['TEFF'],vmin=tempcut-rangeTeff,vmax=tempcut+rangeTeff,
            alpha=0.5,label='low-$\\alpha$')
    plt.scatter(df_plot_ha['Age'],df_plot_ha['M_H'],c=df_plot_ha['TEFF'],vmin=tempcut-rangeTeff,vmax=tempcut+rangeTeff,
            alpha=0.5,label='high-$\\alpha$')
    
    plt.ylabel('Fe')
    plt.xlabel('Age [Gyr]')
    plt.legend()
 
    # for C/N
    df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha['C_FE']>-100]
    df_plot_ha=df_plot_ha.loc[df_plot_ha['N_FE']>-100]
    df_plot_ha=df_plot_ha.reset_index(drop=True)
    
    df_plot_la=df_abun_st_la.loc[df_abun_st_la['C_FE']>-100]
    df_plot_la=df_plot_la.loc[df_plot_la['N_FE']>-100]
    df_plot_la=df_plot_la.reset_index(drop=True)
    
    plt.subplot(5,4,20)
    plt.scatter(df_plot_la['Age'],df_plot_la['C_FE']-df_plot_la['N_FE'],
             c=df_plot_la['TEFF'],alpha=0.5,vmin=tempcut-rangeTeff,vmax=tempcut+rangeTeff,label='low-$\\alpha$')
    plt.scatter(df_plot_ha['Age'],df_plot_ha['C_FE']-df_plot_ha['N_FE'],
             c=df_plot_ha['TEFF'],alpha=0.5,vmin=tempcut-rangeTeff,vmax=tempcut+rangeTeff,label='high-$\\alpha$')
    
    plt.ylabel('[C/N]')
    plt.xlabel('Age [Gyr]')
    plt.legend()


    plt.tight_layout()


    
def plotraw(df_abun_st_ha,df_abun_st_la):
    abun_apog=['M_H','C_FE','N_FE','O_FE','NA_FE','MG_FE','AL_FE','SI_FE','S_FE','K_FE','CA_FE','TI_FE',
           'V_FE','MN_FE','NI_FE','P_FE','CR_FE','CO_FE','RB_FE']

    abun_apog_err=[i+'_ERR' for i in abun_apog]

    abund={'C_FE':'[CI/','MG_FE':'[MgI/','AL_FE':'[AlI/','SI_FE':'[SiI/','S_FE':'[ScI/',
       'CA_FE':'[CaI/','TI_FE':'[TiI/','CR_FE':'[CrI/','MN_FE':'[MnI/','CO_FE':'[CoI/',
       'NI_FE':'[NiI/','CU_FE':'[CuI/','O_FE':'[O/','NA_FE':'[Na/','K_FE':'[K/','V_FE':'[V/',
      'P_FE':'[P/','RB_FE':'[Rb/','N_FE':'[N/','M_H':'[Fe/'}
    
    lim=[-0.5,0.5]
    limx=[0,15]
    plt.figure(figsize=(22.5,25))
    for i in range(1,len(abun_apog)):
        df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha[abun_apog[i]]>-100]
        df_plot_ha=df_plot_ha.reset_index(drop=True)
    
        df_plot_la=df_abun_st_la.loc[df_abun_st_la[abun_apog[i]]>-100]
        df_plot_la=df_plot_la.reset_index(drop=True)
    
        plt.subplot(5,4,i+1)
        plt.errorbar(df_plot_la['Age'],df_plot_la[abun_apog[i]],yerr=df_plot_la[abun_apog[i]+'_ERR'],fmt='bo',alpha=0.5,label='low-$\\alpha$')
        plt.errorbar(df_plot_ha['Age'],df_plot_ha[abun_apog[i]],yerr=df_plot_ha[abun_apog[i]+'_ERR'],fmt='ro',alpha=0.5,label='high-$\\alpha$')
    
    
        plt.ylabel(abund[abun_apog[i]]+'Fe]')
        plt.xlabel('Age [Gyr]')
        plt.ylim(lim)
        plt.xlim(limx)
        plt.legend()
    
    # For Fe
    df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha['M_H']>-100]
    df_plot_ha=df_plot_ha.reset_index(drop=True)
    
    df_plot_la=df_abun_st_la.loc[df_abun_st_la['M_H']>-100]
    df_plot_la=df_plot_la.reset_index(drop=True)
    
    plt.subplot(5,4,1)
    plt.errorbar(df_plot_la['Age'],df_plot_la['M_H'],fmt='bo',alpha=0.5,label='low-$\\alpha$')
    plt.errorbar(df_plot_ha['Age'],df_plot_ha['M_H'],fmt='ro',alpha=0.5,label='high-$\\alpha$')
    
    plt.ylabel('Fe')
    plt.xlabel('Age [Gyr]')
    plt.legend()
    plt.ylim(lim)
    plt.xlim(limx)
 
    # for C/N
    df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha['C_FE']>-100]
    df_plot_ha=df_plot_ha.loc[df_plot_ha['N_FE']>-100]
    df_plot_ha=df_plot_ha.reset_index(drop=True)
    
    df_plot_la=df_abun_st_la.loc[df_abun_st_la['C_FE']>-100]
    df_plot_la=df_plot_la.loc[df_plot_la['N_FE']>-100]
    df_plot_la=df_plot_la.reset_index(drop=True)
    
    plt.subplot(5,4,20)
    plt.errorbar(df_plot_la['Age'],df_plot_la['C_FE']-df_plot_la['N_FE'],
             fmt='bo',alpha=0.5,label='low-$\\alpha$')
    plt.errorbar(df_plot_ha['Age'],df_plot_ha['C_FE']-df_plot_ha['N_FE'],
             fmt='ro',alpha=0.5,label='high-$\\alpha$')
    
    plt.ylabel('[C/N]')
    plt.xlabel('Age [Gyr]')
    plt.legend()
    plt.ylim(lim)
    plt.xlim(limx)


    plt.tight_layout()
    

def plotagedis(df_abun_st_ha,df_abun_st_la,agebin,start,end):
    abun_apog=['M_H','C_FE','N_FE','O_FE','NA_FE','MG_FE','AL_FE','SI_FE','S_FE','K_FE','CA_FE','TI_FE',
           'V_FE','MN_FE','NI_FE','P_FE','CR_FE','CO_FE','RB_FE']

    abun_apog_err=[i+'_ERR' for i in abun_apog]

    abund={'C_FE':'[CI/','MG_FE':'[MgI/','AL_FE':'[AlI/','SI_FE':'[SiI/','S_FE':'[ScI/',
       'CA_FE':'[CaI/','TI_FE':'[TiI/','CR_FE':'[CrI/','MN_FE':'[MnI/','CO_FE':'[CoI/',
       'NI_FE':'[NiI/','CU_FE':'[CuI/','O_FE':'[O/','NA_FE':'[Na/','K_FE':'[K/','V_FE':'[V/',
      'P_FE':'[P/','RB_FE':'[Rb/','N_FE':'[N/','M_H':'[Fe/'}
    
    size=int(np.ceil((end-start)/agebin))
    age_binned=[(2*j+1)*agebin/2+start for j in range(size)]

    ## running median parameters
    agewindow=0.5
    slidwindow=0.1

    # **************************************** #
    plt.figure(figsize=(22.5,25))
    for i in range(1,len(abun_apog)):
        clear_output(wait = True)
        df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha[abun_apog[i]]>-100]
        df_plot_ha=df_plot_ha.loc[df_plot_ha['Age']<10]
        df_plot_ha=df_plot_ha.loc[df_plot_ha['Age_err']<10]
        df_plot_ha=df_plot_ha.reset_index(drop=True)

        df_plot_la=df_abun_st_la.loc[df_abun_st_la[abun_apog[i]]>-100]
        df_plot_la=df_plot_la.loc[df_plot_la['Age']<10]
        df_plot_la=df_plot_la.loc[df_plot_la['Age_err']<10]
        df_plot_la=df_plot_la.reset_index(drop=True)


        df_plot_la=df_plot_la.sort_values(by=['Age'])
        df_plot_ha=df_plot_ha.sort_values(by=['Age'])



        df_plot_la['med_age'],df_plot_la['med_abun']=movingMed_time(df_plot_la['Age'].values,
                                                                    df_plot_la[abun_apog[i]].values,
                                                                    agewindow,slidwindow)
        df_plot_ha['med_age'],df_plot_ha['med_abun']=movingMed_time(df_plot_ha['Age'].values,
                                                                    df_plot_ha[abun_apog[i]].values,
                                                                    agewindow,slidwindow)

        df_plot_la['med_abun']=gaussian_filter(df_plot_la['med_abun'], sigma=10)
        df_plot_ha['med_abun']=gaussian_filter(df_plot_ha['med_abun'], sigma=10)

        df_plot_la_new=df_plot_la.dropna(subset=['med_age','med_abun']).reset_index(drop=True)
        df_plot_ha_new=df_plot_ha.dropna(subset=['med_age','med_abun']).reset_index(drop=True)

        
        plt.subplot(5,4,i+1)


        if len(df_plot_la['Age'])!=0:   
            print(abun_apog[i])

            dispersions_la=[]
            dispersions_ha=[]
            dispersions_la_err=[]
            dispersions_ha_err=[]
            for j in range(size):
                agemin=j*agebin+start
                agemax=(j+1)*agebin+start

                new_df_inbin_la=df_plot_la.loc[df_plot_la['Age']>agemin]
                new_df_inbin_la=new_df_inbin_la.loc[new_df_inbin_la['Age']<agemax]

                new_df_inbin_ha=df_plot_ha.loc[df_plot_ha['Age']>agemin]
                new_df_inbin_ha=new_df_inbin_ha.loc[new_df_inbin_ha['Age']<agemax]


                tot_la,disp_la=calc_disp(new_df_inbin_la,abun_apog[i])
                tot_ha,disp_ha=calc_disp(new_df_inbin_ha,abun_apog[i])

                dispersions_la.append(disp_la)
                dispersions_ha.append(disp_ha)
                
            
                dispersions_la_err.append(disp_la/np.sqrt(len(new_df_inbin_la)))
                dispersions_ha_err.append(disp_ha/np.sqrt(len(new_df_inbin_ha)))

            plt.errorbar(age_binned,dispersions_la,yerr=dispersions_la_err,
                         fmt='b-o',label='low-$\\alpha$')
            plt.errorbar(age_binned,dispersions_ha,yerr=dispersions_ha_err,
                         fmt='r-o',label='high-$\\alpha$')

        plt.ylabel(abund[abun_apog[i]]+'Fe] dispersion')
        plt.xlabel('Age bin center [Gyr]')



        #plt.legend()

    # For Fe
    df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha['M_H']>-100]
    df_plot_ha=df_plot_ha.loc[df_plot_ha['Age_err']<10]
    df_plot_ha=df_plot_ha.reset_index(drop=True)

    df_plot_la=df_abun_st_la.loc[df_abun_st_la['M_H']>-100]
    df_plot_la=df_plot_la.loc[df_plot_la['Age_err']<10]
    df_plot_la=df_plot_la.reset_index(drop=True)

    df_plot_la=df_plot_la.sort_values(by=['Age'])
    df_plot_ha=df_plot_ha.sort_values(by=['Age'])

    df_plot_la['med_age'],df_plot_la['med_abun']=movingMed_time(df_plot_la['Age'].values,
                                                                df_plot_la['M_H'].values,
                                                                agewindow,slidwindow)
    df_plot_ha['med_age'],df_plot_ha['med_abun']=movingMed_time(df_plot_ha['Age'].values,
                                                                df_plot_ha['M_H'].values,
                                                                agewindow,slidwindow)

    df_plot_la['med_abun']=gaussian_filter(df_plot_la['med_abun'], sigma=10)
    df_plot_ha['med_abun']=gaussian_filter(df_plot_ha['med_abun'], sigma=10)

    plt.subplot(5,4,1)
    dispersions_la=[]
    dispersions_ha=[]
    dispersions_la_err=[]
    dispersions_ha_err=[]
    clear_output(wait = True)
    print('FE_H')
    for j in range(size):
        agemin=j*agebin+start
        agemax=(j+1)*agebin+start

        new_df_inbin_la=df_plot_la.loc[df_plot_la['Age']>agemin]
        new_df_inbin_la=new_df_inbin_la.loc[new_df_inbin_la['Age']<agemax]

        new_df_inbin_ha=df_plot_ha.loc[df_plot_ha['Age']>agemin]
        new_df_inbin_ha=new_df_inbin_ha.loc[new_df_inbin_ha['Age']<agemax]


        tot_la,disp_la=calc_disp(new_df_inbin_la,'M_H')
        tot_ha,disp_ha=calc_disp(new_df_inbin_ha,'M_H')

        if len(new_df_inbin_la)==0:
            range_la=0
        else:
            range_la=max(new_df_inbin_la['M_H'])-min(new_df_inbin_la['M_H'])
        if len(new_df_inbin_ha)==0:
            range_ha=0
        else:
            range_ha=max(new_df_inbin_ha['M_H'])-min(new_df_inbin_ha['M_H'])
                    
        dispersions_la_err.append(range_la/np.sqrt(len(new_df_inbin_la)))
        dispersions_ha_err.append(range_ha/np.sqrt(len(new_df_inbin_ha)))
 
        dispersions_la.append(disp_la)
        dispersions_ha.append(disp_ha)

    plt.errorbar(age_binned,dispersions_la,yerr=dispersions_la_err,
                         fmt='b-o',label='low-$\\alpha$')
    plt.errorbar(age_binned,dispersions_ha,yerr=dispersions_ha_err,
                         fmt='r-o',label='high-$\\alpha$')

    plt.ylabel('Fe dispersion')
    plt.xlabel('Age bin center [Gyr]')
    #plt.legend()

    # for C/N
    df_plot_ha=df_abun_st_ha.loc[df_abun_st_ha['C_FE']>-100]
    df_plot_ha=df_plot_ha.loc[df_plot_ha['N_FE']>-100]
    df_plot_ha['C_N']=df_plot_ha['C_FE']-df_plot_ha['N_FE']
    df_plot_ha['C_N_ERR']=np.sqrt(df_plot_ha['C_FE_ERR']**2.+df_plot_ha['N_FE_ERR']**2.)
    df_plot_ha=df_plot_ha.loc[df_plot_ha['Age_err']<10]
    df_plot_ha=df_plot_ha.reset_index(drop=True)

    df_plot_la=df_abun_st_la.loc[df_abun_st_la['C_FE']>-100]
    df_plot_la=df_plot_la.loc[df_plot_la['N_FE']>-100]
    df_plot_la['C_N']=df_plot_la['C_FE']-df_plot_la['N_FE']
    df_plot_la['C_N_ERR']=np.sqrt(df_plot_la['C_FE_ERR']**2.+df_plot_la['N_FE_ERR']**2.)
    df_plot_la=df_plot_la.loc[df_plot_la['Age_err']<10]
    df_plot_la=df_plot_la.reset_index(drop=True)

    df_plot_la=df_plot_la.sort_values(by=['Age'])
    df_plot_ha=df_plot_ha.sort_values(by=['Age'])    

    df_plot_la['med_age'],df_plot_la['med_abun']=movingMed_time(df_plot_la['Age'].values,
                                                                df_plot_la['C_N'].values,
                                                                agewindow,slidwindow)
    df_plot_ha['med_age'],df_plot_ha['med_abun']=movingMed_time(df_plot_ha['Age'].values,
                                                                df_plot_ha['C_N'].values,
                                                                agewindow,slidwindow)

    df_plot_la['med_abun']=gaussian_filter(df_plot_la['med_abun'], sigma=10)
    df_plot_ha['med_abun']=gaussian_filter(df_plot_ha['med_abun'], sigma=10)

    clear_output(wait = True)
    plt.subplot(5,4,20)
    dispersions_la=[]
    dispersions_ha=[]
    dispersions_la_err=[]
    dispersions_ha_err=[]
    clear_output(wait = True)
    print('C_N')
    for j in trange(size):
        agemin=j*agebin+start
        agemax=(j+1)*agebin+start

        new_df_inbin_la=df_plot_la.loc[df_plot_la['Age']>agemin]
        new_df_inbin_la=new_df_inbin_la.loc[new_df_inbin_la['Age']<agemax]

        new_df_inbin_ha=df_plot_ha.loc[df_plot_ha['Age']>agemin]
        new_df_inbin_ha=new_df_inbin_ha.loc[new_df_inbin_ha['Age']<agemax]


        tot_la,disp_la=calc_disp(new_df_inbin_la,'C_N')
        tot_ha,disp_ha=calc_disp(new_df_inbin_ha,'C_N')

        if len(new_df_inbin_la)==0:
            range_la=0
        else:
            range_la=max(new_df_inbin_la['C_N'])-min(new_df_inbin_la['C_N'])
        if len(new_df_inbin_ha)==0:
            range_ha=0
        else:
            range_ha=max(new_df_inbin_ha['C_N'])-min(new_df_inbin_ha['C_N'])
            
        dispersions_la_err.append(range_la/np.sqrt(len(new_df_inbin_la)))
        dispersions_ha_err.append(range_ha/np.sqrt(len(new_df_inbin_ha)))
        
        dispersions_la.append(disp_la)
        dispersions_ha.append(disp_ha)

    plt.errorbar(age_binned,dispersions_la,yerr=dispersions_la_err,
                         fmt='b-o',label='low-$\\alpha$')
    plt.errorbar(age_binned,dispersions_ha,yerr=dispersions_ha_err,
                         fmt='r-o',label='high-$\\alpha$')

    plt.ylabel('[C/N] dispersion')
    plt.xlabel('Age bin center [Gyr]')
    #plt.legend()

    plt.tight_layout()
    plt.savefig('age_disp_as.png')

    
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

def getave_age(age,feh,jz=False,avefunc=np.mean,Nbin=50):
    febins=np.linspace(-0.8,0.5,Nbin)
    
    meanage=np.zeros(Nbin-1)
    meanage_err=np.zeros(Nbin-1)
    meanfeh=np.zeros(Nbin-1)
    starnumb=np.zeros(Nbin-1)
    if jz==True:
        meanjz=np.zeros(Nbin-1)
    for i in trange(Nbin-1):
        binlow=febins[i]
        binhi=febins[i+1]
        #print(binlow,binhi)

        m=(feh>=binlow)&(feh<=binhi)
        
        if sum(m)==0:
            continue
        meanfeh[i]=(binlow+binhi)/2
        #print(sum(m))
        
        meanage[i]=avefunc(age[m])
        meanage_err[i]=(np.std(age[m]))
        starnumb[i]=sum(m)
    return meanfeh,meanage,meanage_err,starnumb

def importage(agecode,dist=True):
    """
    agecode can be "LAMOST", "GALAH", or "ALL"
    """
    if agecode=="LAMOST":
        if dist:
            return pd.read_pickle('/Users/yl4331/Desktop/CreateDataTable/cannonages/allLAMOST_cut_dist.pkl')
        if not dist:
            return pd.read_pickle('/Users/yl4331/Desktop/CreateDataTable/cannonages/allLAMOST_cut.pkl')
    elif agecode=="GALAH":
            return pd.read_pickle('/Users/yl4331/Desktop/CreateDataTable/cannonages/galahages_cut.pkl')
    elif agecode=="ALL":
        if not dist:
            return pd.read_pickle('/Users/yl4331/Desktop/CreateDataTable/cannonages/allages_cut.pkl')
        if dist:
            return pd.read_pickle('/Users/yl4331/Desktop/CreateDataTable/cannonages/allages_cut_dist.pkl')

def importcm(cm1, cm2, cm3=False, cm4=False):
    """
    cm1, cm2 can be "LAMOST", "APOGEE", "RAVE", "TESS", "Kepler_prot", "Kepler_nonprot", "Kepler_all",
    "APOGEE", "GALAH"
    """
    allfiles=glob.glob('/Users/yl4331/Desktop/CreateDataTable/Data/*_cm_*.pkl')
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
        return readfits('/Users/yl4331/Desktop/CreateDataTable/Data/LAMOST-dr5v3-gaia-edr3-coords-distances.fits')
    elif datatype=="LAMOSTkin":
        return readfits('/Users/yl4331/Desktop/CreateDataTable/Data/LAMOST-dr5v3-gaiaEDR3_orbit.fits')
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
def getcolor(vrange,cm=plt.cm.get_cmap('jet')):
    cmap = cm
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

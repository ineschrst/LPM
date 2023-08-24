#Funktionen die man in Optimisierung aktivieren kann 
import pymc as pm
import matplotlib.pyplot as plt 
import arviz as az
import numpy as np
import os
import pandas as pd
from models import exp,shapefree,BinWerte2,IGMix,shapefreeTTD,exponentialMix,inverseGaussianMix,inverseGaussianMixvary,inverseGaussianMixvary1,inverseGaussianMixvary2,exponentialMixvary,exponentialMixvary1,exponentialMixvary2, exponentialMixTTD,inverseGaussianMixTTD
from input_functions import InputWerte

directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'data') #weil unterordner
DTTDM_results_Tano = os.path.join(ordner, 'DTTDM_age_class_results.xlsx')


#prior plot: plotten die genutzten prior gesampelt -> samples=500
def prior_plt():
    prior = pm.sample_prior_predictive(return_inferencedata=False)
    az.plot_dist(prior['age1'],label='age1',color='red')
    az.plot_dist(prior['age2'],label='age2',color='blue')
    az.plot_dist(prior['ratio'],label='ratio',color='green')
    plt.legend()
    #prior_plot.suptitle('Prior Distributions')
    plt.title('Prior')
    plt.show()

def prior_obs_plt_viola(c_Viola,well_num):
    prior = pm.sample_prior_predictive(return_inferencedata=False)
    az.plot_dist(prior['obsC14'],label='obsC14',color='green')
    az.plot_dist(prior['obsAr'],label='obsAr',color='red')
    az.plot_dist(prior['obsC11'],label='obsCFC11',color='blue')
    az.plot_dist(prior['obsC12'],label='obsCFC12',color='purple')
    az.plot_dist(int(np.round(c_Viola[well_num][0])),color='green')
    az.plot_dist(int(np.round(c_Viola[well_num][1])),color='red')
    az.plot_dist(int(np.round(c_Viola[well_num][2])),color='blue')
    az.plot_dist(int(np.round(c_Viola[well_num][3])),color='purple')
    plt.legend()
    plt.title('Prior observed values and observed values')
    plt.show()

def prior_obs_plt(Tracer,c,ar,C11,C12,he,h): 
    prior = pm.sample_prior_predictive(return_inferencedata=False)
    if 'C14' in Tracer:
        az.plot_dist(prior['obsC14'],label='obsC14',color='green')
        az.plot_dist(int(np.round(c)),color='green')
    if 'Ar39' in Tracer:
        az.plot_dist(prior['obsAr'],label='obsAr',color='red')
        az.plot_dist(int(np.round(ar)),color='red')
    if 'CFC11' in Tracer:
        az.plot_dist(prior['obsC11'],label='obsCFC11',color='blue')
        az.plot_dist(int(np.round(C11)),color='blue')
    if 'CFC12' in Tracer:
        az.plot_dist(prior['obsC12'],label='obsCFC12',color='purple')
        az.plot_dist(int(np.round(C12)),color='purple')
    if '4He' in Tracer:
        az.plot_dist(prior['obs4He'],label='obs4He',color='salmon')
        az.plot_dist(int(np.round(he)),color='salmon')
    if '3H' in Tracer:
        az.plot_dist(prior['obs3H'],label='obs3H',color='indigo')
        az.plot_dist(int(np.round(h)),color='indigo')
    
    plt.legend()
    plt.title('Prior observed values and observed values')
    plt.show()

def posterior_obs_plt_viola(c_Viola,well_num,trace):
    posterior = pm.sample_posterior_predictive(trace, return_inferencedata=False)
    az.plot_dist(posterior['obsC14'],color='green',label='C14')
    az.plot_dist(int(np.round(c_Viola[well_num][0])),color='green')
    az.plot_dist(posterior['obsAr'],color='red',label='Argon')
    az.plot_dist(int(np.round(c_Viola[well_num][1])),color='red')
    az.plot_dist(posterior['obsC11'],color='blue',label='CFC11')
    az.plot_dist(int(np.round(c_Viola[well_num][2])),color='blue')
    az.plot_dist(posterior['obsC12'],label='CFC12',color='purple')
    az.plot_dist(int(np.round(c_Viola[well_num][3])),color='purple')
    plt.title('Posterior and observed Values')
    plt.legend()
    plt.show()
    return posterior

def posterior_obs_plt(Tracer,c,ar,C11,C12,he,h,trace):
    posterior = pm.sample_posterior_predictive(trace, return_inferencedata=False)
    if 'C14' in Tracer:
        az.plot_dist(posterior['obsC14'],label='obsC14',color='green')
        az.plot_dist(int(np.round(c)),color='green')
    if 'Ar39' in Tracer:
        az.plot_dist(posterior['obsAr'],label='obsAr',color='red')
        az.plot_dist(int(np.round(ar)),color='red')
    if 'CFC11' in Tracer:
        az.plot_dist(posterior['obsC11'],label='obsCFC11',color='blue')
        az.plot_dist(int(np.round(C11)),color='blue')
    if 'CFC12' in Tracer:
        az.plot_dist(posterior['obsC12'],label='obsCFC12',color='purple')
        az.plot_dist(int(np.round(C12)),color='purple')
    if '4He' in Tracer:
        az.plot_dist(posterior['obs4He'],label='obs4He',color='salmon')
        az.plot_dist(int(np.round(he)),color='salmon')
    if '3H' in Tracer:
        az.plot_dist(posterior['obs3H'],label='obs3H',color='indigo')
        az.plot_dist(int(np.round(h)),color='indigo')

    plt.title('Posterior and observed Values')
    plt.legend()
    plt.show()
    return posterior

def sigma(map_estimate,a_Viola,well_num,summary):
    sigmaage1=abs((map_estimate['age1'].item()-a_Viola[well_num][0])/summary['age1'].sel(metric='sd').item())
    sigmaage2=abs((map_estimate['age2'].item()-a_Viola[well_num][1])/summary['age2'].sel(metric='sd').item())
    sigmaratio=abs((map_estimate['ratio'].item()-a_Viola[well_num][2])/summary['ratio'].sel(metric='sd').item())
    print("Sigma deviation (map and sd summary) age1=",sigmaage1,"age2=",sigmaage2,"ratio=",sigmaratio)

def sigma2(a_Viola,well_num,summary):
    sigmaage1=abs((summary['age1'].sel(metric='mean').item()-a_Viola[well_num][0])/summary['age1'].sel(metric='sd').item())
    sigmaage2=abs((summary['age2'].sel(metric='mean').item()-a_Viola[well_num][1])/summary['age2'].sel(metric='sd').item())
    sigmaratio=abs((summary['ratio'].sel(metric='mean').item()-a_Viola[well_num][2])/summary['ratio'].sel(metric='sd').item())
    print("Sigma deviation (only summary) age1=",sigmaage1,"age2=",sigmaage2,"ratio=",sigmaratio)

def sigmashape(map_estimate,a_shapefreeViola,well_num,summary):
    sigmaratio1=abs((map_estimate['ratio1'].item()-a_shapefreeViola[well_num][0])/summary['ratio1'].sel(metric='sd').item())
    sigmaratio2=abs((map_estimate['ratio2'].item()-a_shapefreeViola[well_num][1])/summary['ratio2'].sel(metric='sd').item())
    ratio3=1-map_estimate['ratio1'].item()-map_estimate['ratio2'].item()
    err3=np.sqrt(summary['ratio1'].sel(metric='sd').item()**2+summary['ratio1'].sel(metric='sd').item()**2)
    sigmaratio3=abs((ratio3-a_shapefreeViola[well_num][2])/err3)
    print("Sigma deviation (map and sd summary) ratio1=",sigmaratio1,"ratio2=",sigmaratio2,"ratio3=",sigmaratio3)

def sigma2shape(a_shapefreeViola,well_num,summary):
    sigmaratio1=abs((summary['ratio1'].sel(metric='mean').item()-a_shapefreeViola[well_num][0])/summary['ratio1'].sel(metric='sd').item())
    sigmaratio2=abs((summary['ratio2'].sel(metric='mean').item()-a_shapefreeViola[well_num][1])/summary['ratio2'].sel(metric='sd').item())
    ratio3=1-summary['ratio1'].sel(metric='mean').item()-summary['ratio2'].sel(metric='mean').item()
    err3=np.sqrt(summary['ratio1'].sel(metric='sd').item()**2+summary['ratio2'].sel(metric='sd').item()**2)
    sigmaratio3=abs((ratio3-a_shapefreeViola[well_num][2])/err3)
    print("Sigma deviation (map and sd summary) ratio1=",sigmaratio1,"ratio2=",sigmaratio2,"ratio3=",sigmaratio3)

def varying_plot(trace,model,c_Viola,sig,well_num,Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,peclet):
    summary=az.summary(trace,fmt='xarray')
    age1=np.arange(start=1,stop=500,step=1)
    age2=np.arange(start=50,stop=10000,step=10)
    ratios=np.arange(start=0,stop=1,step=0.01)
    if model=='Inverse Gaussian Mix':
        c_calcc=inverseGaussianMix(summary['age1'].sel(metric='mean').item(),summary['age2'].sel(metric='mean').item(),summary['ratio'].sel(metric='mean').item(),InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess),t_max,peclet,Anzahl_Tracer,Tracer)
        c_calca=inverseGaussianMixvary1(age1,summary['age2'].sel(metric='mean').item(),summary['ratio'].sel(metric='mean').item(),InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess),t_max,peclet,Anzahl_Tracer,Tracer)
        c_calcag=inverseGaussianMixvary2(summary['age1'].sel(metric='mean').item(),age2,summary['ratio'].sel(metric='mean').item(),InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess),t_max,peclet,Anzahl_Tracer,Tracer)
        c_calcr=inverseGaussianMixvary(summary['age1'].sel(metric='mean').item(),summary['age2'].sel(metric='mean').item(),ratios,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess),t_max,peclet,Anzahl_Tracer,Tracer)
    if model=='Exponential Mix':
        c_calcc=exponentialMix(summary['age1'].sel(metric='mean').item(),summary['age2'].sel(metric='mean').item(),summary['ratio'].sel(metric='mean').item(),InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess),t_max,Anzahl_Tracer,Tracer)
        c_calca=exponentialMixvary1(age1,summary['age2'].sel(metric='mean').item(),summary['ratio'].sel(metric='mean').item(),InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess),t_max,Anzahl_Tracer, Tracer) 
        c_calcag=exponentialMixvary2(summary['age1'].sel(metric='mean').item(),age2,summary['ratio'].sel(metric='mean').item(),InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess),t_max,Anzahl_Tracer, Tracer)  
        c_calcr=exponentialMixvary(summary['age1'].sel(metric='mean').item(),summary['age2'].sel(metric='mean').item(),ratios,ratios,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess),t_max,Anzahl_Tracer, Tracer)
    
    C14_Ar_plot=plt.errorbar(c_calcc[0],c_calcc[1],xerr=0.01*c_calcc[0],yerr=0.01*c_calcc[1],label='c_calc')
    C14_Ar_plot=plt.plot(c_calca[0],c_calca[1],label='c_calc with varying age1')
    C14_Ar_plot=plt.plot(c_calcag[0],c_calcag[1],label='c_calc with varying age2')
    C14_Ar_plot=plt.plot(c_calcr[0],c_calcr[1],label='c_calc with varying ratio')
    C14_Ar_plot=plt.errorbar(c_Viola[well_num][0],c_Viola[well_num][1],xerr=sig[well_num][0],yerr=sig[well_num][0],label='observed values')
    sa=np.round(summary['age1'].sel(metric='mean').item())
    saa=np.round(summary['age2'].sel(metric='mean').item())
    sr=np.round(summary['ratio'].sel(metric='mean').item(),2)
    message=f'c_calc is calculated using age1={sa} age2={saa} and the ratio {sr}'

    age1_values = (1,10,100,400)
    for age1_value in age1_values:
        age1_index = np.where(age1 == age1_value)[0][0]
        plt.text(c_calca[0][age1_index], c_calca[1][age1_index], f'{age1_value} years', ha='left', va='center')
    age2_values = (50,100,1000,9000)
    for age2_value in age2_values:
        age2_index = np.where(age2 == age2_value)[0][0]
        plt.text(c_calcag[0][age2_index], c_calcag[1][age2_index], f'{age2_value} years', ha='left', va='center')
    ratio_values=(0.1,0.5,0.9)
    for ratio_value in ratio_values:
        ratio_index = np.where(ratios == ratio_value)[0][0]
        plt.text(c_calcr[0][ratio_index], c_calcr[1][ratio_index], f'{ratio_value}', ha='left', va='center')

    plt.scatter([], [], color="w", alpha=0, label=message)
    plt.legend()
    plt.show()

#Konvergenz Tool: wie gut die Optimisierung konvergiert (vllt mehrere Abstufungen)
def convergence_extended(trace,summary,iterations):
    print('----------------------------------------------------------------------------------------')
    #plots to view the convergence
    az.plot_trace(trace)
    print('The Trace Plot should have chains that show the same curve on the left and random behavior on the right side')
    az.plot_autocorr(trace)
    print('The Autocorrelation Plot can have values bigger than the grey zone but should pendel into the grey zone')
    az.plot_ess(trace,kind='evolution')
    print('The ESS should have the two curves, bulk and tail, with both of them rising approximately linear')
    az.plot_mcse(trace)
    print('The MCSE Plot should vary around a constant value, without displaying other clear curves')
    plt.show()

    #r_hat should be 1.05 or smaller
    print('R_hat shows the difference between the different chains')
    r_hat=summary['age1'].sel(metric='r_hat').item()
    if r_hat>1.05:
        print("There might be problem with the convergence in age1 r_hat=",np.round(r_hat,4))
    r_hat2=summary['age2'].sel(metric='r_hat').item()
    if r_hat2>1.05:
        print("There might be problem with the convergence in age2 r_hat=",np.round(r_hat2,4))
    r_hat3=summary['ratio'].sel(metric='r_hat').item()
    if r_hat3>1.05:
        print("There might be problem with the convergence in ratio r_hat=",np.round(r_hat3,4))

    #MCSE/posterior sd < 10%
    print('MSCE is the Monte Carlo Standart Error')
    msce=summary['age1'].sel(metric='mcse_mean').item()
    sd=summary['age1'].sel(metric='sd').item()
    if msce/sd > 0.1:
        print('There might be problem with the convergence in age1 msce/sd=',msce/sd)
    msce2=summary['age2'].sel(metric='mcse_mean').item()
    sd2=summary['age2'].sel(metric='sd').item()
    if msce2/sd2 > 0.1:
        print('There might be problem with the convergence in age2 msce/sd=',msce2/sd2)
    msce3=summary['ratio'].sel(metric='mcse_mean').item()
    sd3=summary['ratio'].sel(metric='sd').item()
    if msce3/sd3 > 0.1:
        print('There might be problem with the convergence in ratio msce/sd=',msce3/sd3)

    #ESS/iteration > 10%
    print('ESS is the effective sample size')
    ess=summary['age1'].sel(metric='ess_bulk').item()
    if ess/iterations < 0.1:
        print('There might be problem with the convergence in age1 ess/iterations=',ess/iterations)
    ess2=summary['age2'].sel(metric='ess_bulk').item()
    if ess2/iterations < 0.1:
        print('There might be problem with the convergence in age2 ess/iterations=',ess2/iterations)
    ess3=summary['ratio'].sel(metric='ess_bulk').item()
    if ess3/iterations < 0.1:
        print('There might be problem with the convergence in ratio ess/iterations=',ess3/iterations)

    print('----------------------------------------------------------------------------------------')

def convergence(trace, summary,iterations):
    print('----------------------------------------------------------------------------------------')
    az.plot_trace(trace)
    plt.show()
    
    #r_hat should be 1.05 or smaller
    r_hat=summary['age1'].sel(metric='r_hat').item()
    r_hat2=summary['age2'].sel(metric='r_hat').item()
    r_hat3=summary['ratio'].sel(metric='r_hat').item()
    if r_hat>1.05 or r_hat2>1.05 or r_hat3>1.05:
        print("There might be problem with the convergence in age1 r_hat=",r_hat, 'or age2 r_hat=',r_hat2,'or ratio r_hat=',r_hat3)

    #MCSE/posterior sd < 10%
    msce=summary['age1'].sel(metric='mcse_mean').item()
    sd=summary['age1'].sel(metric='sd').item()
    msce2=summary['age2'].sel(metric='mcse_mean').item()
    sd2=summary['age2'].sel(metric='sd').item()
    msce3=summary['ratio'].sel(metric='mcse_mean').item()
    sd3=summary['ratio'].sel(metric='sd').item()
    if msce/sd > 0.1 or msce2/sd2 > 0.1 or msce3/sd3 > 0.1:
        print('There might be problem with the convergence in age1 msce/sd=',msce/sd, 'or age2 msce/sd=',msce2/sd2,'or ratio msce/sd=',msce3/sd3)

    #ESS/iteration > 10%
    ess=summary['age1'].sel(metric='ess_bulk').item()
    ess2=summary['age2'].sel(metric='ess_bulk').item()
    ess3=summary['ratio'].sel(metric='ess_bulk').item()
    if ess/iterations < 0.1 or ess2/iterations < 0.1 or ess3/iterations < 0.1:
        print('There might be problem with the convergence in age1 ess/iterations=',ess/iterations, 'or age2 ess/iterations=',ess2/iterations,'or ratio ess/iterations=',ess3/iterations)

    print('----------------------------------------------------------------------------------------')

def convergence_extended_shapefree(trace,summary,iterations):
    print('----------------------------------------------------------------------------------------')
    #plots to view the convergence
    az.plot_trace(trace)
    print('The Trace Plot should have chains that show the same curve on the left and random behavior on the right side')
    az.plot_autocorr(trace)
    print('The Autocorrelation Plot can have values bigger than the grey zone but should pendel into the grey zone')
    az.plot_ess(trace,kind='evolution')
    print('The ESS should have the two curves, bulk and tail, with both of them rising approximately linear')
    az.plot_mcse(trace)
    print('The MCSE Plot should vary around a constant value, without displaying other clear curves')
    plt.show()

    #r_hat should be 1.05 or smaller
    print('R_hat shows the difference between the different chains')
    r_hat=summary['ratio1'].sel(metric='r_hat').item()
    if r_hat>1.05:
        print("There might be problem with the convergence in ratio1 r_hat=",r_hat)
    r_hat2=summary['ratio2'].sel(metric='r_hat').item()
    if r_hat2>1.05:
        print("There might be problem with the convergence in ratio2 r_hat=",r_hat2)

    #MCSE/posterior sd < 10%
    print('MSCE is the Monte Carlo Standart Error')
    msce=summary['ratio1'].sel(metric='mcse_mean').item()
    sd=summary['ratio1'].sel(metric='sd').item()
    if msce/sd > 0.1:
        print('There might be problem with the convergence in ratio1 msce/sd=',msce/sd)
    msce2=summary['ratio2'].sel(metric='mcse_mean').item()
    sd2=summary['ratio2'].sel(metric='sd').item()
    if msce2/sd2 > 0.1:
        print('There might be problem with the convergence in ratio2 msce/sd=',msce2/sd2)

    #ESS/iteration > 10%
    print('ESS is the effective sample size')
    ess=summary['ratio1'].sel(metric='ess_bulk').item()
    if ess/iterations < 0.1:
        print('There might be problem with the convergence in ratio1 ess/iterations=',ess/iterations)
    ess2=summary['ratio2'].sel(metric='ess_bulk').item()
    if ess2/iterations < 0.1:
        print('There might be problem with the convergence in ratio2 ess/iterations=',ess2/iterations)

    print('----------------------------------------------------------------------------------------')

def convergence_shapefree(trace,summary,iterations):
    print('----------------------------------------------------------------------------------------')
    az.plot_trace(trace)
    plt.show()
    #r_hat should be 1.05 or smaller
    r_hat=summary['ratio1'].sel(metric='r_hat').item()
    r_hat2=summary['ratio2'].sel(metric='r_hat').item()
    if r_hat>1.05 or r_hat2>1.05:
        print("There might be problem with the convergence in ratio1 r_hat=",r_hat, 'or ratio2 r_hat=',r_hat2)

    #MCSE/posterior sd > 10%
    msce=summary['ratio1'].sel(metric='mcse_mean').item()
    sd=summary['ratio1'].sel(metric='sd').item()
    msce2=summary['ratio2'].sel(metric='mcse_mean').item()
    sd2=summary['ratio2'].sel(metric='sd').item()
    if msce/sd > 0.1 or msce2/sd2 > 0.1:
        print('There might be problem with the convergence in ratio1 msce/sd=',msce/sd, 'or ratio2 msce/sd=',msce2/sd2)

    #ESS/iteration > 10%
    ess=summary['ratio1'].sel(metric='ess_bulk').item()
    ess2=summary['ratio2'].sel(metric='ess_bulk').item()
    if ess/iterations < 0.1 or ess2/iterations < 0.1:
        print('There might be problem with the convergence in ratio1 ess/iterations=',ess/iterations, 'or ratio2 ess/iterations=',ess2/iterations)

    print('----------------------------------------------------------------------------------------')

def convergence_shapefree6(trace,summary,iterations): #for six ratios
    print('----------------------------------------------------------------------------------------')
    az.plot_trace(trace)
    r_hat=True
    ms=True #mcse/sd 
    ess=True #ess/iterations

    plt.show()
    #r_hat should be 1.05 or smaller
    r_hat1=summary['ratio1'].sel(metric='r_hat').item()
    r_hat2=summary['ratio2'].sel(metric='r_hat').item()
    r_hat3=summary['ratio3'].sel(metric='r_hat').item()
    r_hat4=summary['ratio4'].sel(metric='r_hat').item()
    r_hat5=summary['ratio5'].sel(metric='r_hat').item()
    r_hat6=summary['ratio6'].sel(metric='r_hat').item()
    if r_hat1>1.05 or r_hat2>1.05 or r_hat3>1.05 or r_hat4>1.05 or r_hat5>1.05 or r_hat6>1.05:
        print("There might be problem with the convergence in r_hat")
        r_hat=False

    #MCSE/posterior sd < 10%
    msce1=summary['ratio1'].sel(metric='mcse_mean').item()
    sd1=summary['ratio1'].sel(metric='sd').item()
    msce2=summary['ratio2'].sel(metric='mcse_mean').item()
    sd2=summary['ratio2'].sel(metric='sd').item()
    msce3=summary['ratio3'].sel(metric='mcse_mean').item()
    sd3=summary['ratio3'].sel(metric='sd').item()
    msce4=summary['ratio4'].sel(metric='mcse_mean').item()
    sd4=summary['ratio4'].sel(metric='sd').item()
    msce5=summary['ratio5'].sel(metric='mcse_mean').item()
    sd5=summary['ratio5'].sel(metric='sd').item()
    msce6=summary['ratio6'].sel(metric='mcse_mean').item()
    sd6=summary['ratio6'].sel(metric='sd').item()
    if msce1/sd1 > 0.1 or msce2/sd2 > 0.1 or msce3/sd3 > 0.1 or msce4/sd4 > 0.1 or msce5/sd5 > 0.1 or msce6/sd6 > 0.1:
        print('There might be problem with the convergence msce/sd')
        ms=False

    #ESS/iteration > 10%
    ess1=summary['ratio1'].sel(metric='ess_bulk').item()
    ess2=summary['ratio2'].sel(metric='ess_bulk').item()
    ess3=summary['ratio3'].sel(metric='ess_bulk').item()
    ess4=summary['ratio4'].sel(metric='ess_bulk').item()
    ess5=summary['ratio5'].sel(metric='ess_bulk').item()
    ess6=summary['ratio6'].sel(metric='ess_bulk').item()
    if ess1/iterations < 0.1 or ess2/iterations < 0.1 or ess3/iterations < 0.1 or ess4/iterations < 0.1 or ess5/iterations < 0.1 or ess6/iterations < 0.1:
        print('There might be problem with the convergence ess/iterations')
        ess=False

    print('----------------------------------------------------------------------------------------')
    return r_hat,ms,ess

def two_opt(summary,summary2, a_Viola, well_num, iterations1, iterations2,trace,trace2):
    #deviation value
    age1=summary['age1'].sel(metric='mean').item()
    age2=summary['age2'].sel(metric='mean').item()
    ratio=summary['ratio'].sel(metric='mean').item()
    age21=summary2['age1'].sel(metric='mean').item()
    age22=summary2['age2'].sel(metric='mean').item()
    ratio2=summary2['ratio'].sel(metric='mean').item()
    print('The values chainging from the first optimization to the second: age1 from',age1,'years to',age21,'years; age2 from',age2,'years to',age22,'years; ratio from',ratio,'to',ratio2)

    #deviation sigma (only summary sigma)
    sigmaage1=abs((summary['age1'].sel(metric='mean').item()-a_Viola[well_num][0])/summary['age1'].sel(metric='sd').item())
    sigmaage2=abs((summary['age2'].sel(metric='mean').item()-a_Viola[well_num][1])/summary['age2'].sel(metric='sd').item())
    sigmaratio=abs((summary['ratio'].sel(metric='mean').item()-a_Viola[well_num][2])/summary['ratio'].sel(metric='sd').item())
    sigmaage21=abs((summary2['age1'].sel(metric='mean').item()-a_Viola[well_num][0])/summary2['age1'].sel(metric='sd').item())
    sigmaage22=abs((summary2['age2'].sel(metric='mean').item()-a_Viola[well_num][1])/summary2['age2'].sel(metric='sd').item())
    sigmaratio2=abs((summary2['ratio'].sel(metric='mean').item()-a_Viola[well_num][2])/summary2['ratio'].sel(metric='sd').item())
    print("Sigma deviation changing from the first optimization to the second: age1 from",sigmaage1,'to',sigmaage21,"; age2 from",sigmaage2,'to',sigmaage22, "; ratio from",sigmaratio,'to',sigmaratio2)

    #deviation relative error
    rel_age1=summary['age1'].sel(metric='sd').item()/summary['age1'].sel(metric='mean').item()*100
    rel_age2=summary['age2'].sel(metric='sd').item()/summary['age2'].sel(metric='mean').item()*100
    rel_ratio=summary['ratio'].sel(metric='sd').item()/summary['ratio'].sel(metric='mean').item()*100
    rel2_age1=summary2['age1'].sel(metric='sd').item()/summary2['age1'].sel(metric='mean').item()*100
    rel2_age2=summary2['age2'].sel(metric='sd').item()/summary2['age2'].sel(metric='mean').item()*100
    rel2_ratio=summary2['ratio'].sel(metric='sd').item()/summary2['ratio'].sel(metric='mean').item()*100
    print('The relative error from the first to the second optimization: age1 from',rel_age1,'% to ',rel2_age1,'%; age2 from',rel_age2,'% to ',rel2_age2,'%; ratio from',rel_ratio,'% to ',rel2_ratio,'%')

    #convergence
    r_hat1=summary['age1'].sel(metric='r_hat').item()
    r_hat2=summary['age2'].sel(metric='r_hat').item()
    r_hat3=summary['ratio'].sel(metric='r_hat').item()
    r_hat21=summary2['age1'].sel(metric='r_hat').item()
    r_hat22=summary2['age2'].sel(metric='r_hat').item()
    r_hat23=summary2['ratio'].sel(metric='r_hat').item()
    if r_hat21<r_hat1 and r_hat22<r_hat2 and r_hat23<r_hat3:
        print('R_hat improved for all variables')
    elif r_hat21<1.05 and r_hat22<1.05 and r_hat23<1.05:
        print('R_hat<1.05 for all variables in the second optimization')
    msce1=summary['age1'].sel(metric='mcse_mean').item()
    sd1=summary['age1'].sel(metric='sd').item()
    msce2=summary['age2'].sel(metric='mcse_mean').item()
    sd2=summary['age2'].sel(metric='sd').item()
    msce3=summary['ratio'].sel(metric='mcse_mean').item()
    sd3=summary['ratio'].sel(metric='sd').item()
    msce21=summary2['age1'].sel(metric='mcse_mean').item()
    sd21=summary2['age1'].sel(metric='sd').item()
    msce22=summary2['age2'].sel(metric='mcse_mean').item()
    sd22=summary2['age2'].sel(metric='sd').item()
    msce23=summary2['ratio'].sel(metric='mcse_mean').item()
    sd23=summary2['ratio'].sel(metric='sd').item()
    if msce1/sd1<msce21/sd21 and msce2/sd2<msce22/sd22 and msce3/sd3<msce23/sd23:
        print('MSCE over Standart Deviation has improved for all variables')
    elif msce21/sd21 > 0.1 or msce22/sd22 > 0.1 or msce23/sd23 > 0.1:
        print(f'MSCE over Standart Deviation is bigger than 10% for all variables after the second optimization')
    ess1=summary['age1'].sel(metric='ess_bulk').item()
    ess2=summary['age2'].sel(metric='ess_bulk').item()
    ess3=summary['ratio'].sel(metric='ess_bulk').item()
    ess21=summary2['age1'].sel(metric='ess_bulk').item()
    ess22=summary2['age2'].sel(metric='ess_bulk').item()
    ess23=summary2['ratio'].sel(metric='ess_bulk').item()
    if ess1/iterations1<ess21/iterations2 and ess2/iterations1<ess22/iterations2 and ess3/iterations1<ess23/iterations2:
        print('ESS over iterations has improved for all variables')
    elif ess21/iterations2>0.1 and ess22/iterations2>0.1 and ess23/iterations2>0.1:
        print(f'ESS over iterations is bigger than 10% for all variables after the second optimization')

    #plot density of both
    az.plot_density([trace,trace2],data_labels=['first optimization','second optimization'])
    plt.show()
    
def count_tracers(tracers):
    # Define the tracers
    possible_tracers= ['C14','Ar39','CFC11','CFC12','4He','3H','NGT']

    # Initialize the counter
    count = 0

    # Loop through each tracer in the list
    for tracer in possible_tracers:
        # Count the occurrences of the tracer in the input string
        count += tracers.count(tracer)
    return count

def TTD(summary,model,t_max): #display Transit Time Distribution of analytic Model Results
    if model=='Inverse Gaussian Mix':
        inverseGaussianMixTTD(summary['age1'].sel(metric='mean').item(),summary['age2'].sel(metric='mean').item(),summary['ratio'].sel(metric='mean').item(),t_max)
    if model=='Exponential Mix':
        exponentialMixTTD(summary['age1'].sel(metric='mean').item(),summary['age2'].sel(metric='mean').item(),summary['ratio'].sel(metric='mean').item(),t_max)
    return

def deviation_ratios(summary,well):
    if well<19:
        #BW Werte
        name=pd.read_excel(DTTDM_results_Tano,sheet_name='BW',usecols='A',skiprows=well,nrows=1).values
        ratio=pd.read_excel(DTTDM_results_Tano,sheet_name='BW',usecols='B:F',skiprows=well,nrows=1).values
        time=['<100 yr','100-1000 yr','1000-10000 yr','10000-25000 yr','>25000 yr']
    elif well>21 and well<26:
        #BW Werte
        well=well-3
        name=pd.read_excel(DTTDM_results_Tano,sheet_name='BW',usecols='A',skiprows=well,nrows=1).values
        ratio=pd.read_excel(DTTDM_results_Tano,sheet_name='BW',usecols='B:F',skiprows=well,nrows=1).values
        time=['<100 yr','100-1000 yr','1000-10000 yr','10000-25000 yr','>25000 yr']
    elif well>26 and well<41:
        #BW Werte
        well=well-4
        name=pd.read_excel(DTTDM_results_Tano,sheet_name='BW',usecols='A',skiprows=well,nrows=1).values
        ratio=pd.read_excel(DTTDM_results_Tano,sheet_name='BW',usecols='B:F',skiprows=well,nrows=1).values
        time=['<100 yr','100-1000 yr','1000-10000 yr','10000-25000 yr','>25000 yr']
    elif well>40 and well<60:
        #VT Werte
        well=well-41
        name=pd.read_excel(DTTDM_results_Tano,sheet_name='VT',usecols='A',skiprows=well,nrows=1).values
        ratio=pd.read_excel(DTTDM_results_Tano,sheet_name='VT',usecols='B:G',skiprows=well,nrows=1).values
        time=['<70 yr','70-250 yr','250-1000 yr','1000-10000 yr','10000-25000 yr','>25000 yr']
    elif well>59 and well<63:
        #VO Werte
        well=well-60+4
        name=pd.read_excel(DTTDM_results_Tano,sheet_name='HO & VO',usecols='A',skiprows=well,nrows=1).values
        ratio=pd.read_excel(DTTDM_results_Tano,sheet_name='HO & VO',usecols='B:G',skiprows=well,nrows=1).values
        time=['<100 yr','100-300 yr','300-1000 yr','1000-10000 yr','10000-25000 yr','>25000 yr']
    elif well>63 and well<66:
        #VO Werte
        well=well-60+3
        name=pd.read_excel(DTTDM_results_Tano,sheet_name='HO & VO',usecols='A',skiprows=well,nrows=1).values
        ratio=pd.read_excel(DTTDM_results_Tano,sheet_name='HO & VO',usecols='B:G',skiprows=well,nrows=1).values
        time=['<100 yr','100-300 yr','300-1000 yr','1000-10000 yr','10000-25000 yr','>25000 yr']
    elif well>76 and well<81:
        #HO Werte
        well=well-77
        name=pd.read_excel(DTTDM_results_Tano,sheet_name='HO & VO',usecols='A',skiprows=well,nrows=1).values
        ratio=pd.read_excel(DTTDM_results_Tano,sheet_name='HO & VO',usecols='B:G',skiprows=well,nrows=1).values
        time=['<100 yr','100-300 yr','300-1000 yr','1000-10000 yr','10000-25000 yr','>25000 yr']
    elif 18<well<22 or well==26 or well==63 or 65<well<77:
        #missing values
        print("This well doesn't exist in the DTTDM age results data")
        name=0
        time=0
        ratio=0   
    sigmaratio1=np.round((abs((summary['ratio1'].sel(metric='mean').item()-ratio[0][0])/summary['ratio1'].sel(metric='sd').item())),3)
    sigmaratio2=np.round((abs((summary['ratio2'].sel(metric='mean').item()-ratio[0][1])/summary['ratio2'].sel(metric='sd').item())),3)
    sigmaratio3=np.round((abs((summary['ratio3'].sel(metric='mean').item()-ratio[0][2])/summary['ratio3'].sel(metric='sd').item())),3)
    sigmaratio4=np.round((abs((summary['ratio4'].sel(metric='mean').item()-ratio[0][3])/summary['ratio4'].sel(metric='sd').item())),3)
    sigmaratio5=np.round((abs((summary['ratio5'].sel(metric='mean').item()-ratio[0][4])/summary['ratio5'].sel(metric='sd').item())),3)
    sigmaratio6=np.round((abs((summary['ratio6'].sel(metric='mean').item()-ratio[0][5])/summary['ratio6'].sel(metric='sd').item())),3)

    print("Well",name)
    print("Sigma deviation ratio1 =",sigmaratio1,", ratio2 =",sigmaratio2,", ratio3 =",sigmaratio3,", ratio4 =",sigmaratio4,", ratio5 =",sigmaratio5,", ratio6 =",sigmaratio6)
    print("Time Bins",time)
    return sigmaratio1,sigmaratio2,sigmaratio3,sigmaratio4,sigmaratio5,sigmaratio6

def relative_error(ratios,err_ratios):
    relative=err_ratios/ratios
    return relative

def saving_shape(trace,well_name,excel_file_path,ratios,err_ratios,end,start,r_hat,mcse,ess,sigmaratio,relative): #to save the results of PyMC shapefree
    #saves the trace plot
    az.plot_trace(trace)
    figname = "traceplot_compare3_well_{}.pdf".format(well_name[0])
    savefolder='C:/Users/InesChrista/Bachelorarbeit/Grumpy-master/LPM/results/trace_plots' #uni pc
    #savefolder='C:/InesZeug/Bachelorarbeit/Grumpy-master/py/results' #zuhause pc
    path=os.path.join(savefolder,figname)
    plt.savefig(path, format='pdf')
    
    #saves the values to an excel file
    df_results = pd.DataFrame({
        'Well Name': well_name.flatten(),
        'Ratio 1': ratios[0],
        'Ratio 2': ratios[1],
        'Ratio 3': ratios[2],
        'Ratio 4': ratios[3],
        'Ratio 5': ratios[4],
        'Ratio 6': ratios[5],
        'SD Ratio 1': err_ratios[0],
        'SD Ratio 2': err_ratios[1],
        'SD Ratio 3': err_ratios[2],
        'SD Ratio 4': err_ratios[3],
        'SD Ratio 5': err_ratios[4],
        'SD Ratio 6': err_ratios[5],
        'Time': end - start,
        'r_hat': r_hat,
        'mcse': mcse,
        'ess':ess,
        'sigmaratio1':sigmaratio[0],
        'sigmaratio2':sigmaratio[1],
        'sigmaratio3':sigmaratio[2],
        'sigmaratio4':sigmaratio[3],
        'sigmaratio5':sigmaratio[4],
        'sigmaratio6':sigmaratio[5],
        'relative1':relative[0],
        'relative2':relative[1],
        'relative3':relative[2],
        'relative4':relative[3],
        'relative5':relative[4],
        'relative6':relative[5]
    })

    # Write the DataFrame to an Excel file
    with pd.ExcelWriter(excel_file_path, mode='a',if_sheet_exists='overlay') as writer:
        df_results.to_excel(writer, index=False, header=False,startrow=writer.sheets["Sheet1"].max_row)

def plots_priors(ratio1,ratio2,ratio3,ratio4,ratio5,ratio6):
    ratios = [ratio1, ratio2, ratio3, ratio4, ratio5, ratio6]
    for i, ratio in enumerate(ratios):
        if ratio != 0:
            samples = pm.draw(ratio, draws=2000)
            plt.hist(samples, bins=70, density=True, histtype="stepfilled")
            plt.title(f"Prior distribution for ratio{i+1}")
            figname = f"prior_plot_ratio{i+1}.pdf"
            savefolder='C:/Users/InesChrista/Bachelorarbeit/Grumpy-master/LPM/results/Prior_plots' #uni pc
            #savefolder='C:/InesZeug/Bachelorarbeit/Grumpy-master/py/results' #zuhause pc
            path=os.path.join(savefolder,figname)
            plt.savefig(path, format='pdf')
            plt.show()
    return 


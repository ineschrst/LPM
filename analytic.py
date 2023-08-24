#The PyMC analytic model
import time
import pymc as pm
import pymc.math as pmm
import numpy as np
import arviz as az
import os 
import pandas as pd
import matplotlib.pyplot as plt
from models import exp,shapefree,BinWerte2,IGMix,shapefreeTTD,shapefreeTTD2,exponentialMix,inverseGaussianMix,inverseGaussianMixvary,exponentialMixvary,exponentialMixvary1,exponentialMixvary2
from input_functions import InputWerte
from opti_functions import prior_plt,prior_obs_plt,posterior_obs_plt,sigma,sigma2,varying_plot,sigmashape,sigma2shape, convergence, convergence_extended, convergence_extended_shapefree, convergence_shapefree, two_opt,count_tracers,TTD,deviation_ratios, convergence_shapefree6

start=time.time()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Model Setup

t_max=50000
Tracer='Ar39'+'4He'+'3H' #written here should be all the used tracers, all possible tracers are: C14,Ar39,CFC11,CFC12,4He,3H, (NGT missing input function)
Anzahl_Tracer=count_tracers(Tracer)
deep=False #this chooses the accumulation rate for 4He: deep=False means shallow 1.5e-11 ccSTP/g, deep=True means deep 4.75e-11ccSTP/g
model='Inverse Gaussian Mix' #this chooses the model: Inverse Gaussian Mix, Exponential Mix

#the following values are only important for the CFCs
peclet=10 
vogel=True 
temp = 24.0
salinity = 1.0
pressure = 0.955
excess = 2.


#plots
TTDs=True

#the second optimization step
opt_model=False

#convergence tool (convergence, convergence extended)
conv_tool=True
conv_ex_tool=False

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#data netherlands

well=79 #this chooses the well, in the list list_value-2=well
#Which values are what? 0-40 BW, 41-59 VT, 60-62 VO, 77-80 HO (Seppe) -> no comparing for 18<well<22 or well==26 or well==63 or 65<well<77
#Zegge HO (5-13) 81-88

directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'data')
measFile = os.path.join(ordner, 'VT_VO_HO.xlsx')

well_name=pd.read_excel(measFile, usecols='A').values
h=pd.read_excel(measFile, usecols='AG').values #whole rows and each value for each well
err_h=pd.read_excel(measFile, usecols='AH').values
NGT=pd.read_excel(measFile, usecols='U').values
err_NGT=pd.read_excel(measFile, usecols='V').values
he=pd.read_excel(measFile, usecols='W').values
err_he=pd.read_excel(measFile, usecols='X').values
ar=pd.read_excel(measFile, usecols='AE').values #not for every well
err_ar=pd.read_excel(measFile, usecols='AF').values
c=pd.read_excel(measFile,usecols='AC').values #C14
err_c=np.ones_like(c) #error of C14 is not given

well_name=well_name[well]
h=h[well]
err_h=err_h[well]
NGT=NGT[well]
err_NGT=err_NGT[well]
he=he[well]
#err_he=err_he[well]
err_he=[1e-8] #not an error for most samples (use same for DTTDM and here) !!!
ar=ar[well]
err_ar=err_ar[well]
c=c[well]
err_c=err_c[well]
#print(well_name)
#print(h,NGT,he,ar,c)
#print(err_h,err_NGT,err_he,err_ar,err_c)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Optimization and Model Fitting

def pymc_analytic():
    # Create the PyMC3 model
    with pm.Model() as groundwater_model:
        #model parameters as flat priors
        age1=pm.Uniform('age1',lower=1,upper=500)
        age2=pm.Uniform('age2',lower=50,upper=10000)
        ratio = pm.Uniform('ratio', lower=0, upper=1)

        #calculated tracer values depending on the choosen model
        if model=='Inverse Gaussian Mix':
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H=IGMix(age1,age2,ratio,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),t_max,Tracer)
        if model=='Exponential Mix':
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H=exp(age1,age2,ratio,t_max,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),Tracer)
        
        #Likelihood
        if 'C14' in Tracer:
            likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=err_c, observed=c)
        if 'Ar39' in Tracer:
            likelihood_funcAr = pm.Normal('obsAr', mu=c_calcAr, sigma=err_ar, observed=ar)
        if 'CFC11' in Tracer:
            likelihood_funcC11 = pm.Normal('obsC11', mu=c_calcC11, sigma=0, observed=0) #If CFCs are used, the sigma and observed value need to be added
        if 'CFC12' in Tracer:
            likelihood_funcC12 = pm.Normal('obsC12', mu=c_calcC12, sigma=0, observed=0)
        if '4He' in Tracer:
            likelihood_func4He = pm.Normal('obs4He', mu=c_calc4He, sigma=err_he, observed=he)
        if '3H' in Tracer:
            likelihood_func3H = pm.Normal('obs3H', mu=c_calc3H, sigma=err_h, observed=h)
    
        # Perform the sampling
        step=pm.Metropolis() 
        its=50000 #iterations 
        trace=pm.sample(its, tune=1000,step=step, chains=4) #four chains are used to ensure convergence 
        map_estimate=pm.find_MAP()
    
    summary=az.summary(trace,fmt='xarray')
    end=time.time()
    az.plot_trace(trace)
    output_file = 'IGunconverged2.pdf'
    plt.savefig(output_file, format='pdf')
    plt.show()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Convergence Tool   

    if conv_tool==True:
        convergence(trace,summary,its) #also gives trace plot
    if conv_ex_tool==True:
        convergence_extended(trace,summary,its)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Results and Output

    #plots
    az.plot_forest(trace) #good for seeing values+error
    plt.show()

    print("time",end-start)

    print("well sample ID",well_name)
    print(az.summary(trace))
    print("MAP for well", well_name, map_estimate)

    if TTDs==True:
        TTD(summary,model,t_max)
    plt.show()
    return summary, end
    
    
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Second Optimization Step
def pymc_analytic_opt(summary, end):

    start2=time.time() 
    with pm.Model() as opt_model:
        #The piors of the second optimization step are the results of the first optimization step
        age1=pm.Normal('age1',mu=summary['age1'].sel(metric='mean').item(),sigma=summary['age1'].sel(metric='sd').item())
        age2=pm.Normal('age2',mu=summary['age2'].sel(metric='mean').item(),sigma=summary['age2'].sel(metric='sd').item())
        ratio=pm.TruncatedNormal('ratio',mu=summary['ratio'].sel(metric='mean').item(),sigma=summary['ratio'].sel(metric='sd').item(),lower=0,upper=1)

        #calculated tracer values depending on the choosen model
        if model=='Inverse Gaussian Mix':
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H=IGMix(age1,age2,ratio,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),t_max,Tracer)
        if model=='Exponential Mix':
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H=exp(age1,age2,ratio,t_max,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),Tracer)
        
        #Likelihood
        if 'C14' in Tracer:
            likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=err_c, observed=c)
        if 'Ar39' in Tracer:
            likelihood_funcAr = pm.Normal('obsAr', mu=c_calcAr, sigma=err_ar, observed=ar)
        if 'CFC11' in Tracer:
            likelihood_funcC11 = pm.Normal('obsC11', mu=c_calcC11, sigma=0, observed=0)
        if 'CFC12' in Tracer:
            likelihood_funcC12 = pm.Normal('obsC12', mu=c_calcC12, sigma=0, observed=0)
        if '4He' in Tracer:
            likelihood_func4He = pm.Normal('obs4He', mu=c_calc4He, sigma=err_he, observed=he)
        if '3H' in Tracer:
            likelihood_func3H = pm.Normal('obs3H', mu=c_calc3H, sigma=err_h, observed=h)

        #Perform the sampling
        step=pm.Metropolis() 
        its2=5000 #iterations
        trace2=pm.sample(its2, tune=5000,step=step, chains=4)
        map_estimate2=pm.find_MAP()
    
    summary2=az.summary(trace2,fmt='xarray')
    end2=time.time()
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Convergence Tool   
    az.plot_trace(trace2)
    output_file = 'IGunconverged.pdf'
    plt.savefig(output_file, format='pdf')
    plt.show()

    if conv_tool==True:
        convergence(trace2,summary2,its2) #also gives trace plot
    if conv_ex_tool==True:
        convergence_extended(trace2,summary2,its2)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Results and Output

    #plots
    az.plot_forest(trace2) #good for seeing values+error
    plt.show()

    print("time",end2-start2)

    print("well sample ID",well_name)
    print(az.summary(trace2))
    print("MAP2 for well", well_name, map_estimate2)

    print('time for both optimizations',((end2-start2)+(end-start)))

    if TTDs==True:
        TTD(summary2,model,t_max)
    plt.show()
        

            
if __name__ == '__main__':
    summary,end=pymc_analytic() 
    if opt_model==True:
        pymc_analytic_opt(summary,end)      
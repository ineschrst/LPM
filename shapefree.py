#The PyMC shapefree Model
import time
import pymc as pm
import pymc.math as pmm
import numpy as np
import arviz as az
import arviz.labels as azl
import os 
import pandas as pd
import matplotlib.pyplot as plt
from models import exp,shapefree,BinWerte2,IGMix,shapefreeTTD,shapefreeTTD2,exponentialMix,inverseGaussianMix,inverseGaussianMixvary,exponentialMixvary,exponentialMixvary1,exponentialMixvary2
from input_functions import InputWerte
from opti_functions import prior_plt,prior_obs_plt,posterior_obs_plt,sigma,sigma2,varying_plot,sigmashape,sigma2shape, convergence, convergence_extended, convergence_extended_shapefree, convergence_shapefree, two_opt,count_tracers,TTD,deviation_ratios, convergence_shapefree6,saving_shape, relative_error,plots_priors,deviation_ratios2

start=time.time()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Model Setup

#t_grenzen=[0,100,300,1000,10000,25000,50000] #age classes
t_grenzen=[0,100,1000,10000,25000,50000] #age classes BW
t_max= np.max(t_grenzen)
Tracer='C14'+'4He'+'3H'+'NGT' #written here should be all the used tracers, all possible tracers are: C14,Ar39,CFC11,CFC12,4He,3H,NGT 'Ar39'+
Anzahl_Tracer=count_tracers(Tracer)
deep=True #this chooses the accumulation rate for 4He: deep=False means shallow 1.5e-11 ccSTP/g, deep=True means deep 4.75e-11ccSTP/g

#the following values are only important for the CFCs
peclet=10 
vogel=True 
temp = 24.0
salinity = 1.0
pressure = 0.955
excess = 2.

#plots
TTDs=True
Priors=False #plotting Prior Distributions

#sigma
sigmaabweichung=False

#relative error
relative=True

#convergence tool (convergence, convergence extended)
conv_tool=True
conv_ex_tool=False 

#saving the results
save=True

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#data netherlands

well=9 #this chooses the well, in the list list_value-2=well
#Which values are what? 0-40 BW, 41-59 VT, 60-62 VO, 77-80 HO (Seppe) -> no comparing for 18<well<22 or well==26 or well==63 or 65<well<77
#Zegge HO (5-13) 81-88

directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'data') 
measFile = os.path.join(ordner, 'VT_VO_HO.xlsx')
carbonageFile=os.path.join(ordner, 'apparent_ages.xlsx')

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
cage=pd.read_excel(carbonageFile, usecols='C').values #empty for some samples
err_cage=pd.read_excel(carbonageFile, usecols='D').values

well_name=well_name[well]
h=h[well]
err_h=err_h[well]
NGT=NGT[well]
err_NGT=err_NGT[well]
he=he[well]
err_he=err_he[well]
#err_he=0.05*he #no error given, 5% of helium as error
ar=ar[well]
err_ar=err_ar[well]
c=c[well]
err_c=err_c[well]
cage=cage[well]
err_cage=err_cage[well]
#print(well_name)
#print(h,NGT,he,ar,c)
#print(err_h,err_NGT,err_he,err_ar,err_c)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Optimization and Model Fitting 
anzahl_bins=len(t_grenzen)-1
if (anzahl_bins-1)>Anzahl_Tracer: #to check whether enough tracers are being used
    print("This model is not robust")
    print("Bins =",anzahl_bins,"  Tracers =",Anzahl_Tracer,"  min. number of tracers should be",(anzahl_bins-1))

rhoC14,rhoAr, rhoC11,rhoC12,rho4He,rho3H,rhoNGT=BinWerte2(InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),t_grenzen,Tracer,Anzahl_Tracer) #the rho values only depend on the modeled values and don't need to be recalculated for each ratio distribution
#BinWerte2 is ajusted to BW or HO&VO (NGT and 3H input functions are just the bin values)
#print(rhoC14,rhoAr, rhoC11,rhoC12,rho4He,rho3H,rhoNGT)

def pymc_shapefree():
    with pm.Model() as groundwater_shapefreemodel:
        #depending on the number of bins the ratios are choosen as flat priors
        if anzahl_bins==2:
            ratio1=pm.Uniform('ratio1', lower=0, upper=1)
            ratio2=pm.Deterministic('ratio2',(1-ratio1))
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,0,0,0,0,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)
        if anzahl_bins==3:
            ratio1=pm.Uniform('ratio1', lower=0, upper=1)
            ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
            ratio3=pm.Deterministic('ratio3',(1-ratio1-ratio2))
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,ratio3,0,0,0,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)
        if anzahl_bins==4:
            ratio1=pm.Uniform('ratio1', lower=0, upper=1)
            ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
            ratio3=pm.Uniform('ratio3', lower=0, upper=1-ratio1-ratio2)
            ratio4=pm.Deterministic('ratio4',(1-ratio1-ratio2-ratio3))
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,ratio3,ratio4,0,0,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)
        if anzahl_bins==5:
            ratio1=pm.Uniform('ratio1', lower=0, upper=1)
            ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
            ratio3=pm.Uniform('ratio3', lower=0, upper=1-ratio1-ratio2)
            ratio4=pm.Uniform('ratio4', lower=0, upper=1-ratio1-ratio2-ratio3)
            ratio5=pm.Deterministic('ratio5',(1-ratio1-ratio2-ratio3-ratio4))
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,ratio3,ratio4,ratio5,0,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)
        if anzahl_bins==6:
            '''
            ratio1=pm.Uniform('ratio1', lower=0, upper=1)
            ratio2=pm.Uniform('ratio2', lower=0, upper=1)
            ratio3=pm.Uniform('ratio3', lower=0, upper=1)
            ratio4=pm.Uniform('ratio4', lower=0, upper=1)
            ratio5=pm.Uniform('ratio5', lower=0, upper=1)
            ratio6=pm.Deterministic('ratio6',(1-ratio1-ratio2-ratio3-ratio4-ratio5))
            '''
            ratio1=pm.Uniform('ratio1', lower=0, upper=1)
            ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
            ratio3=pm.Uniform('ratio3', lower=0, upper=1-ratio1-ratio2)
            ratio4=pm.Uniform('ratio4', lower=0, upper=1-ratio1-ratio2-ratio3)
            ratio5=pm.Uniform('ratio5', lower=0, upper=1-ratio1-ratio2-ratio3-ratio4)
            ratio6=pm.Deterministic('ratio6',(1-ratio1-ratio2-ratio3-ratio4-ratio5))
            constraint = ratio6 >= 0
            #constraint = (ratio1+ratio2+ratio3+ratio4+ratio5) <= 1
            potential = pm.Potential("ratio6_constraint", pmm.log(pmm.switch(constraint, 1, 0.01)))
            '''
            ratio1=pm.TruncatedNormal('ratio1', mu=0.04, sigma=0.1, lower=0, upper=1)
            ratio2=pm.TruncatedNormal('ratio2', mu=0.52, sigma=0.2, lower=0, upper=1)
            ratio3=pm.TruncatedNormal('ratio3', mu=0.24, sigma=0.15, lower=0, upper=1)
            ratio4=pm.TruncatedNormal('ratio4', mu=0.1, sigma=0.1, lower=0, upper=1)
            ratio5=pm.TruncatedNormal('ratio5', mu=0.1, sigma=0.1, lower=0, upper=1)
            ratio6=pm.Deterministic('ratio6', (1-ratio1-ratio2-ratio3-ratio4-ratio5))
            '''
            c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,ratio3,ratio4,ratio5,ratio6,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)

        #Prior Plots
        if Priors==True:
            if anzahl_bins == 6:
                plots_priors(ratio1, ratio2, ratio3, ratio4, ratio5, ratio6)
            elif anzahl_bins == 1:
                plots_priors(ratio1, 0, 0, 0, 0, 0)
            elif anzahl_bins == 2:
                plots_priors(ratio1, ratio2, 0, 0, 0, 0)
            elif anzahl_bins == 3:
                plots_priors(ratio1, ratio2, ratio3, 0, 0, 0)
            elif anzahl_bins == 4:
                plots_priors(ratio1, ratio2, ratio3, ratio4, 0, 0)
            elif anzahl_bins == 5:
                plots_priors(ratio1, ratio2, ratio3, ratio4, ratio5, 0)


        #Likelihood 
        if 'C14' in Tracer:
            likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=err_cage, observed=cage) #C14 age likelihood
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
        if 'NGT' in Tracer:
            likelihood_funcNGT = pm.Normal('obsNGT', mu=c_calcNGT, sigma=err_NGT, observed=NGT)
        

        # Perform the sampling
        step=pm.NUTS(target_accept=0.95)
        its=1000 #iterations
        traceshape=pm.sample(its,step=step, tune=1000,chains=4) #four chains are used to ensure convergence 
        best=pm.find_MAP() #MAP Maximum a posterior (best is not really the correct name)
        pm.sample_posterior_predictive(traceshape,extend_inferencedata=True)
        summ=az.summary(traceshape,group='posterior_predictive',round_to=10)

        
    end=time.time() #time measurement. The output, saving and plots aren't included

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Convergence Tool 

    summaryshape=az.summary(traceshape,fmt='xarray')
    if conv_tool==True:
        r_hat,mcse,ess=convergence_shapefree6(traceshape,summaryshape,its) #ajusted to BW
        print(r_hat,mcse,ess)
    if conv_ex_tool==True:
        convergence_extended_shapefree(traceshape,summaryshape,its)


#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Results and Output

    print("well sample ID",well_name)
    print('time', end-start)
    print('MAP:', best)
    summaryshape=az.summary(traceshape,fmt='wide')
    print(az.summary(traceshape))

    #ratios=np.array([summaryshape.loc['ratio1','mean'],summaryshape.loc['ratio2','mean'],summaryshape.loc['ratio3','mean'],summaryshape.loc['ratio4','mean'],summaryshape.loc['ratio5','mean'],(1-summaryshape.loc['ratio1','mean']-summaryshape.loc['ratio2','mean']-summaryshape.loc['ratio3','mean']-summaryshape.loc['ratio4','mean']-summaryshape.loc['ratio5','mean'])]) #ajusted to the number of bins
    #err_ratios=np.array([summaryshape.loc['ratio1','sd'],summaryshape.loc['ratio2','sd'],summaryshape.loc['ratio3','sd'],summaryshape.loc['ratio4','sd'],summaryshape.loc['ratio5','sd'],summaryshape.loc['ratio6','sd']])
    ratios=np.array([summaryshape.loc['ratio1','mean'],summaryshape.loc['ratio2','mean'],summaryshape.loc['ratio3','mean'],summaryshape.loc['ratio4','mean'],(1-summaryshape.loc['ratio1','mean']-summaryshape.loc['ratio2','mean']-summaryshape.loc['ratio3','mean']-summaryshape.loc['ratio4','mean'])]) #ajusted to the number of bins
    err_ratios=np.array([summaryshape.loc['ratio1','sd'],summaryshape.loc['ratio2','sd'],summaryshape.loc['ratio3','sd'],summaryshape.loc['ratio4','sd'],summaryshape.loc['ratio5','sd']]) #BW


    summaryshape=az.summary(traceshape,fmt='xarray')
    if sigmaabweichung==True:
        sigmaratio1,sigmaratio2,sigmaratio3,sigmaratio4,sigmaratio5,sigmaratio6=deviation_ratios2(best,summaryshape,well) #sigmaratio6=0 for BW
        sigmaratio=[sigmaratio1,sigmaratio2,sigmaratio3,sigmaratio4,sigmaratio5,sigmaratio6]
        print("Sigma Deviation of the DTTDM ratios to the PyMC results:",sigmaratio)
        #sigma=0
    
    if relative==True:
        relative_err=relative_error(ratios,err_ratios) 
        print("The relative Error of the ratios:",relative_err)

    if TTDs==True:
        shapefreeTTD(ratios,t_grenzen,well_name)
        shapefreeTTD2(ratios,err_ratios,t_grenzen,well_name)

    if save==True:
        sigmaratio1,sigmaratio2,sigmaratio3,sigmaratio4,sigmaratio5,sigmaratio6=deviation_ratios2(best,summaryshape,well) #sigmaratio6=0 for BW
        relative_err=relative_error(ratios,err_ratios)
        sigmaratio=[sigmaratio1,sigmaratio2,sigmaratio3,sigmaratio4,sigmaratio5,sigmaratio6]
        #sigmaratio=0
        #obsAr=summ.loc['obsAr[0]','mean']
        obsAr=0
        obsC14=summ.loc['obsC14[0]','mean']
        #obsC14=0
        obsNGT=summ.loc['obsNGT[0]','mean']
        obs3H=summ.loc['obs3H[0]','mean']
        obs4He=summ.loc['obs4He[0]','mean']
        #sd_obsAr=summ.loc['obsAr[0]','sd']
        sd_obsAr=0
        sd_obsC14=summ.loc['obsC14[0]','sd']
        #sd_obsC14=0
        sd_obsNGT=summ.loc['obsNGT[0]','sd']
        sd_obs3H=summ.loc['obs3H[0]','sd']
        sd_obs4He=summ.loc['obs4He[0]','sd']
        excel_file_path = 'C:/Users/InesChrista/Bachelorarbeit/Grumpy-master/LPM/results/shapefree.xlsx' #the excel file needs to be closed while running the code
        #excel_file_path = 'C:/InesZeug/Bachelorarbeit/Grumpy-master/py/results/results2.xlsx' #zuhause
        saving_shape(traceshape,well_name,excel_file_path,ratios,err_ratios,end,start,r_hat,mcse,ess,sigmaratio,relative_err,deep,best,obsAr,obsNGT,obsC14,obs3H,obs4He,sd_obsAr,sd_obsNGT,sd_obsC14,sd_obs3H,sd_obs4He) #changed for BW
        print("Results saved in Excel file:", excel_file_path)



if __name__ == '__main__':
    pymc_shapefree()
    
#Optimisierung von a
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
#from comparing import deviation_ratios
start=time.time()
#_____________________________________________________________________________________________________________________________________________________________________________
#werte
#für Input und Optimisierung
t_grenzen=[0,100,300,1000,10000,25000,100000] #mit NGT -> 6 bins sonst geht es momentan nicht (NGT Werte für Bins [0,100,300,1000,10000,25000,100000])
timestep1=1
timestep2=1
t_max1=100
t_max2=100000
t_max=100000
Tracer='Ar39'+'4He'+'3H'+'NGT' #C14,Ar39,CFC11,CFC12,4He,3H,NGT
Anzahl_Tracer=count_tracers(Tracer)
model='Inverse Gaussian Mix' #Inverse Gaussian Mix, Exponential Mix
peclet=10 #siehe viola
vogel=True 
temp = 24.0
salinity = 1.0
pressure = 0.955
excess = 2.
deep=False
#_____________________________________________________________________________________________________________________________________________________________________________
#plots
prior_plot=False
prior_obs_plot=False
posterior_obs_plot=False
vary_plot=False
TTDs=True

#_____________________________________________________________________________________________________________________________________________________________________________
#sigma
sigmaabweichung=False

#_____________________________________________________________________________________________________________________________________________________________________________
#zweite Optimisierung 
opt_model=False
opt_compare=False #nur wenn opt model True, sonst sinnlos
opt_shape=False

#_____________________________________________________________________________________________________________________________________________________________________________
#convergence tool (convergence, convergence extended)
conv_tool=True
conv_ex_tool=False


#_____________________________________________________________________________________________________________________________________________________________________________
#rhoC14,rhoAr, rhoC11,rhoC12=BinWerte2(InputWerte(Tracer,Anzahl_Tracer,vogel,t_max1,t_max2,timestep1,timestep2,temp,salinity,pressure,excess),t_grenzen,Tracer,Anzahl_Tracer)
a_Viola=([10,1650,0.63],[10,19000,0.24],[10,13000,0.51],[10,4700,0.71],[10,4700,0.63],[10,8000,0.51],[10,4700,0.67]) #Brunnen 1,19,20,25,33,58,66
#young MRT, old MRT, ratio r     MRT in years  
a_shapefreeViola=([0.63,0,0.37],[0.24,0,0.76], [0.51,0,0.49], [0.71,0,0.29], [0.63,0,0.37], [0.51,0,0.49], [0.67,0,0.33]) #ratios für das shapefree Model -> keine Daten bei Viola können nur geraten werden
c_Viola=([78.30720715,66, 1.6, 1],[29.57595373, 34, 0.6, 1.1],[53.75564384, 94, 1.3, 0.9],[73.89712479, 82, 1.8, 1.4],[71.12855882, 61, 1.6, 1.1],[60.02607455, 71, 1.3, 0.64],[72.74457494, 67, 1.7, 0.9]) #for wells 1,19,20,25,33,58,66; well 1 c_calcViola[0][:] (also in Zeilen geordnet)
sig=([0.17,12,0.2,0.1],[0.11,6,0.1,0.1],[0.13,11,0.2,0.1],[0.17,10,0.2,0.1],[0.16,9,0.2,0.1],[0.15,13,0.2,0.05],[0.16,12,0.2,0.1])
well_num=3
Brunnen=np.array([1,19,20,25,33,58,66])

#_____________________________________________________________________________________________________________________________________________________________________________
#Messwerte Niederland
well=77 #in the list list_value-2=well
#Which values are what? 0-40 BW, 41-59 VT, 60-62 VO, 77-80 HO (Seppe) -> no comparing for 18<well<22 or well==26 or well==63 or 65<well<77
#Zegge HO (5-13) 81-88
directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'data') #weil unterordner
measFile = os.path.join(ordner, 'VT_VO_HO.xlsx')
ageFile=os.path.join(ordner, 'Broers_data.xlsx')

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
err_he=[1e-8] #not an error for most samples (use same for DTTDM and here)
ar=ar[well]
err_ar=err_ar[well]
c=c[well]
err_c=err_c[well]
#print(well_name)
#print(h,NGT,he,ar,c)
#print(err_h,err_NGT,err_he,err_ar,err_c)
#_____________________________________________________________________________________________________________________________________________________________________________

normal=False
if normal==True:
    # Create the PyMC3 model
    with pm.Model() as groundwater_model:
        age1=pm.Uniform('age1',lower=1,upper=500)
        age2=pm.Uniform('age2',lower=50,upper=10000)
        ratio = pm.Uniform('ratio', lower=0, upper=1)

    if __name__ == '__main__':
        with groundwater_model:
        
            if model=='Inverse Gaussian Mix':
                c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H=IGMix(age1,age2,ratio,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),t_max1,t_max2,timestep1,timestep2,Tracer)
            if model=='Exponential Mix':
                c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H=exp(age1,age2,ratio,t_max,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),Tracer)
            
            #Likelihood
            if 'C14' in Tracer:
                #likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=sig[well_num][0], observed=c_Viola[well_num][0])
                likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=err_c, observed=c)
            if 'Ar39' in Tracer:
                #likelihood_funcAr = pm.Normal('obsAr', mu=c_calcAr, sigma=sig[well_num][1], observed=c_Viola[well_num][1])
                likelihood_funcAr = pm.Normal('obsAr', mu=c_calcAr, sigma=err_ar, observed=ar)
            if 'CFC11' in Tracer:
                likelihood_funcC11 = pm.Normal('obsC11', mu=c_calcC11, sigma=sig[well_num][2], observed=c_Viola[well_num][2])
            if 'CFC12' in Tracer:
                likelihood_funcC12 = pm.Normal('obsC12', mu=c_calcC12, sigma=sig[well_num][3], observed=c_Viola[well_num][3])
            if '4He' in Tracer:
                likelihood_func4He = pm.Normal('obs4He', mu=c_calc4He, sigma=err_he, observed=he)
            if '3H' in Tracer:
                likelihood_func3H = pm.Normal('obs3H', mu=c_calc3H, sigma=err_h, observed=h)
        
            # Perform the sampling
            step=pm.Metropolis() #otherwise very slow 
            its=1000 #iterations 
            trace=pm.sample(its, tune=1000,step=step, chains=4)
            map_estimate=pm.find_MAP()
            #likk=pm.compute_log_likelihood(trace)

            #plots
            if prior_plot==True:
                prior_plt()
                print("prior plot")
            if prior_obs_plot==True:
                prior_obs_plt(c_Viola,well_num)
                print("prior obs plot")
            if posterior_obs_plot==True:
                posterior_obs_plt(c_Viola,well_num,trace)
                print("posterior obs plot")

            if vary_plot==True:
                varying_plot(trace,model,c_Viola,sig,well_num,Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,peclet)

            summary=az.summary(trace,fmt='xarray')
        

        end=time.time()
        print("time",end-start)
        print(az.summary(trace))

        print("MAP for Brunnen", Brunnen[well_num], map_estimate)
        if sigmaabweichung==True:  
            sigma(map_estimate,a_Viola,well_num,summary)
            sigma2(a_Viola,well_num,summary)
        
        az.plot_forest(trace) #good for seeing values+error

        if conv_tool==True:
            convergence(trace,summary,its) #also gives trace plot
        if conv_ex_tool==True:
            convergence_extended(trace,summary,its)

        if TTDs==True:
            TTD(summary,model,t_max)

        plt.show()
        
        
#_____________________________________________________________________________________________________________________________________________________________________________  
        if opt_model==True:
            start2=time.time() 
            with pm.Model() as opt_model:
                age1=pm.Normal('age1',mu=summary['age1'].sel(metric='mean').item(),sigma=summary['age1'].sel(metric='sd').item())
                age2=pm.Normal('age2',mu=summary['age2'].sel(metric='mean').item(),sigma=summary['age2'].sel(metric='sd').item())
                ratio=pm.TruncatedNormal('ratio',mu=summary['ratio'].sel(metric='mean').item(),sigma=summary['ratio'].sel(metric='sd').item(),lower=0,upper=1)

                if model=='Inverse Gaussian Mix':
                    c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He=IGMix(age1,age2,ratio,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),t_max1,t_max2,timestep1,timestep2,Tracer)
                if model=='Exponential Mix':
                    c_calcC14,c_calcAr,c_calcC11,c_calcC12=exp(age1,age2,ratio,t_max,InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),Tracer)
                #Likelihood
                if 'C14' in Tracer:
                    #likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=sig[well_num][0], observed=c_Viola[well_num][0])
                    likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=err_c, observed=c)
                if 'Ar39' in Tracer:
                    #likelihood_funcAr = pm.Normal('obsAr', mu=c_calcAr, sigma=sig[well_num][1], observed=c_Viola[well_num][1])
                    likelihood_funcAr = pm.Normal('obsAr', mu=c_calcAr, sigma=err_ar, observed=ar)
                if 'CFC11' in Tracer:
                    likelihood_funcC11 = pm.Normal('obsC11', mu=c_calcC11, sigma=sig[well_num][2], observed=c_Viola[well_num][2])
                if 'CFC12' in Tracer:
                    likelihood_funcC12 = pm.Normal('obsC12', mu=c_calcC12, sigma=sig[well_num][3], observed=c_Viola[well_num][3])
                if '4He' in Tracer:
                    likelihood_func4He = pm.Normal('obs4He', mu=c_calc4He, sigma=err_he, observed=he)
                
                step=pm.Metropolis() 
                its2=5000
                trace2=pm.sample(its2, tune=5000,step=step, chains=4)
                map_estimate2=pm.find_MAP()

                summary2=az.summary(trace2,fmt='xarray')
                if vary_plot==True:
                    varying_plot(trace2,model,c_Viola,sig,well_num,Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,peclet)

                print(az.summary(trace2))
            end2=time.time()
            print("MAP2 for Brunnen", Brunnen[well_num], map_estimate2)
            if sigmaabweichung==True:
                sigma(map_estimate2,a_Viola,well_num,summary2)
                sigma2(a_Viola,well_num,summary2)

            
            if conv_tool==True:
                convergence(trace2,summary2,its2)
            if conv_ex_tool==True:
                convergence_extended(trace2,summary2,its2)
            plt.show()
            print('time for both optimizations',((end2-start2)+(end-start)))

            if opt_compare==True:
                two_opt(summary, summary2,a_Viola, well_num,its, its2, trace, trace2)
        


#_____________________________________________________________________________________________________________________________________________________________________________
shape=True
anzahl_bins=6
if (anzahl_bins-1)>Anzahl_Tracer:
    print("This model is not robust")
    print("Bins =",anzahl_bins,"  Tracers =",Anzahl_Tracer,"  min. number of tracers should be",(anzahl_bins-1))
 #number of bins -> free parameters bins-1, shouldn't be more than Tracers 
if shape==True:
    startshape=time.time()
    #shapefree Model
    rhoC14,rhoAr, rhoC11,rhoC12,rho4He,rho3H,rhoNGT=BinWerte2(InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),t_grenzen,Tracer,Anzahl_Tracer)
    if __name__ == '__main__':
        with pm.Model() as groundwater_shapefreemodel:
            if anzahl_bins==2:
                ratio1=pm.Uniform('ratio1', lower=0, upper=1)
                #ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
                ratio2=pm.Deterministic('ratio2',(1-ratio1))
                c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,0,0,0,0,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)
            if anzahl_bins==3:
                ratio1=pm.Uniform('ratio1', lower=0, upper=1)
                ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
                #ratio3=pm.Uniform('ratio3', lower=0, upper=1-ratio1-ratio2)
                ratio3=pm.Deterministic('ratio3',(1-ratio1-ratio2))
                c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,ratio3,0,0,0,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)
            if anzahl_bins==4:
                ratio1=pm.Uniform('ratio1', lower=0, upper=1)
                ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
                ratio3=pm.Uniform('ratio3', lower=0, upper=1-ratio1-ratio2)
                #ratio4=pm.Uniform('ratio4', lower=0, upper=1-ratio1-ratio2-ratio3)
                ratio4=pm.Deterministic('ratio4',(1-ratio1-ratio2-ratio3))
                c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,ratio3,ratio4,0,0,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)
            if anzahl_bins==5:
                ratio1=pm.Uniform('ratio1', lower=0, upper=1)
                ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
                ratio3=pm.Uniform('ratio3', lower=0, upper=1-ratio1-ratio2)
                ratio4=pm.Uniform('ratio4', lower=0, upper=1-ratio1-ratio2-ratio3)
                #ratio5=pm.Uniform('ratio5', lower=0, upper=1-ratio1-ratio2-ratio3-ratio4)
                ratio5=pm.Deterministic('ratio5',(1-ratio1-ratio2-ratio3-ratio4))
                c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,ratio3,ratio4,ratio5,0,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)
            if anzahl_bins==6:
                ratio1=pm.Uniform('ratio1', lower=0, upper=1)
                ratio2=pm.Uniform('ratio2', lower=0, upper=1-ratio1)
                ratio3=pm.Uniform('ratio3', lower=0, upper=1-ratio1-ratio2)
                ratio4=pm.Uniform('ratio4', lower=0, upper=1-ratio1-ratio2-ratio3)
                ratio5=pm.Uniform('ratio5', lower=0, upper=1-ratio1-ratio2-ratio3-ratio4)
                #ratio6=1-ratio1-ratio2-ratio3-ratio4-ratio5
                ratio6=pm.Deterministic('ratio6',(1-ratio1-ratio2-ratio3-ratio4-ratio5))
                #ratio6=pm.Uniform('ratio6', lower=0, upper=1-ratio1-ratio2-ratio3-ratio4-ratio5)
                c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT=shapefree(ratio1,ratio2,ratio3,ratio4,ratio5,ratio6,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT,Tracer,t_grenzen)

            #Likelihood 
            if 'C14' in Tracer:
                likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=err_c, observed=c) #viola: sigma=sig[well_num][0],observed=c_Viola[well_num][0]
            if 'Ar39' in Tracer:
                likelihood_funcAr = pm.Normal('obsAr', mu=c_calcAr, sigma=err_ar, observed=ar) #sigma=sig[well_num][1], observed=c_Viola[well_num][1]
            if 'CFC11' in Tracer:
                likelihood_funcC11 = pm.Normal('obsC11', mu=c_calcC11, sigma=sig[well_num][2], observed=c_Viola[well_num][2])
            if 'CFC12' in Tracer:
                likelihood_funcC12 = pm.Normal('obsC12', mu=c_calcC12, sigma=sig[well_num][3], observed=c_Viola[well_num][3])
            if '4He' in Tracer:
                likelihood_func4He = pm.Normal('obs4He', mu=c_calc4He, sigma=err_he, observed=he)
            if '3H' in Tracer:
                likelihood_func3H = pm.Normal('obs3H', mu=c_calc3H, sigma=err_h, observed=h)
            if 'NGT' in Tracer:
                likelihood_funcNGT = pm.Normal('obsNGT', mu=c_calcNGT, sigma=err_NGT, observed=NGT)
           
    
            # Perform the sampling
            step=pm.NUTS(target_accept=0.95)
            its3=1000
            traceshape=pm.sample(its3,step=step, tune=1000,chains=4)
            best=pm.find_MAP()

        print("well sample id",well_name)
        endshape1=time.time()
        print('time shapefree',(endshape1-start))
        summaryshape=az.summary(traceshape)
        print(az.summary(traceshape))
        #ratios=np.array([summaryshape.loc['ratio1','mean'],summaryshape.loc['ratio2','mean'],summaryshape.loc['ratio3','mean'],(1-summaryshape.loc['ratio1','mean']-summaryshape.loc['ratio2','mean']-summaryshape.loc['ratio3','mean'])])
        #ratios=np.array([summaryshape.loc['ratio1','mean'],(1-summaryshape.loc['ratio1','mean'])])
        #ratios=np.array([summaryshape.loc['ratio1','mean'],summaryshape.loc['ratio2','mean'],summaryshape.loc['ratio3','mean'],summaryshape.loc['ratio4','mean'],(1-summaryshape.loc['ratio1','mean']-summaryshape.loc['ratio2','mean']-summaryshape.loc['ratio3','mean']-summaryshape.loc['ratio4','mean'])])
        ratios=np.array([summaryshape.loc['ratio1','mean'],summaryshape.loc['ratio2','mean'],summaryshape.loc['ratio3','mean'],summaryshape.loc['ratio4','mean'],summaryshape.loc['ratio5','mean'],(1-summaryshape.loc['ratio1','mean']-summaryshape.loc['ratio2','mean']-summaryshape.loc['ratio3','mean']-summaryshape.loc['ratio4','mean']-summaryshape.loc['ratio5','mean'])])
        err_ratios=np.array([summaryshape.loc['ratio1','sd'],summaryshape.loc['ratio2','sd'],summaryshape.loc['ratio3','sd'],summaryshape.loc['ratio4','sd'],summaryshape.loc['ratio5','sd'],summaryshape.loc['ratio6','sd']])
        bins=np.arange(anzahl_bins)+1
        shapefreeTTD(ratios,t_grenzen,well_name)
        shapefreeTTD2(ratios,err_ratios,t_grenzen,well_name)
        print(ratios)
        print(best)
        az.plot_trace(traceshape)
        plt.title("Trace Plot Well {}".format(well_name[0]))
        figname = "traceplot_well_{}.png".format(well_name[0])
        savefolder='C:/Users/InesChrista/Bachelorarbeit/Grumpy-master/LPM/results/trace_plots'
        path=os.path.join(savefolder,figname)
        plt.savefig(path)
        plt.show()
        summaryshape=az.summary(traceshape,fmt='xarray')
        if sigmaabweichung==True:
            sigmashape(best,a_shapefreeViola,well_num,summaryshape)
            sigma2shape(a_shapefreeViola,well_num,summaryshape)
        if conv_tool==True:
            convergence_shapefree(traceshape,summaryshape,its3)
            r_hat,mcse,ess=convergence_shapefree6(traceshape,summaryshape,its3)
            print(r_hat,mcse,ess)
        if conv_ex_tool==True:
            convergence_extended_shapefree(traceshape,summaryshape,its3)
        print("shapefree opt1 time",endshape1-startshape)

        deviation_ratios(summaryshape,well)

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
            'Time': endshape1 - start,
            'r_hat': r_hat,
            'mcse': mcse,
            'ess':ess
        })

        # Specify the path where you want to save the Excel file
        excel_file_path = 'C:/Users/InesChrista/Bachelorarbeit/Grumpy-master/LPM/results/results2.xlsx' #das excel file muss währenddessen zu sein, sonst läuft es nicht!

        # Write the DataFrame to an Excel file
        with pd.ExcelWriter(excel_file_path, mode='a',if_sheet_exists='overlay') as writer:
            df_results.to_excel(writer, index=False, header=False,startrow=writer.sheets["Sheet1"].max_row)

        #df_results.to_excel(excel_file_path, index=False, header=False,startrow=1)

        print("Results saved in Excel file:", excel_file_path)
#_____________________________________________________________________________________________________________________________________________________________________________
        if opt_shape==True: #needs to be updated if used
            with pm.Model() as opt_shapefree:
                ratio1=pm.Normal('ratio1', mu=summaryshape['ratio1'].sel(metric='mean').item(),sigma=summaryshape['ratio1'].sel(metric='sd').item())
                ratio2=pm.Normal('ratio2', mu=summaryshape['ratio2'].sel(metric='mean').item(),sigma=summaryshape['ratio2'].sel(metric='sd').item())
                ratio3=1-ratio1-ratio2
                c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H=shapefree(ratio1,ratio2,ratio3,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,Tracer)

                #Likelihood 
                if 'C14' in Tracer:
                    likelihood_funcC14 = pm.Normal('obsC14', mu=c_calcC14, sigma=sig[well_num][0], observed=c_Viola[well_num][0])
                if 'Ar39' in Tracer:
                    likelihood_funcAr = pm.Normal('obsAr', mu=c_calcAr, sigma=sig[well_num][1], observed=c_Viola[well_num][1])
                if 'CFC11' in Tracer:
                    likelihood_funcC11 = pm.Normal('obsC11', mu=c_calcC11, sigma=sig[well_num][2], observed=c_Viola[well_num][2])
                if 'CFC12' in Tracer:
                    likelihood_funcC12 = pm.Normal('obsC12', mu=c_calcC12, sigma=sig[well_num][3], observed=c_Viola[well_num][3])
            

                # Perform the sampling
                step=pm.NUTS(target_accept=0.95)
                its4=1000
                traceshape2=pm.sample(its4, tune=1000, step=step, chains=4)
                best2=pm.find_MAP()

            endshape2=time.time()
            summaryshape2=az.summary(traceshape2)
            print(az.summary(traceshape2))
            ratios2=np.array([summaryshape2.loc['ratio1','mean'],summaryshape2.loc['ratio2','mean'],(1-summaryshape2.loc['ratio1','mean']-summaryshape2.loc['ratio2','mean'])])
            bins2=np.array([1,2,3])
            shapefreeTTD(ratios2,bins2)

            print(ratios2)
            print(best2)
            az.plot_trace(traceshape2)
    
            plt.show()
            summaryshape2=az.summary(traceshape2,fmt='xarray')
            if sigmaabweichung==True:
                sigmashape(best2,a_shapefreeViola,well_num,summaryshape2)
                sigma2shape(a_shapefreeViola,well_num,summaryshape2)
            if conv_tool==True:
                convergence_shapefree(traceshape2,summaryshape2,its4)
            if conv_ex_tool==True:
                convergence_extended_shapefree(traceshape2,summaryshape2,its4)
            print("shapefree both opt time",endshape2-startshape)
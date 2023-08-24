#tool to compare results of DTTDM and shapefree pymc
#first maybe comparison between excel and DTTDM and then excel and shapfree pymc
import numpy as np
import os
import pandas as pd 

directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'data') #weil unterordner
DTTDM_results_Tano = os.path.join(ordner, 'DTTDM_age_class_results.xlsx')

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


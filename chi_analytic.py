#chi or chi^2 test for analytic results

import numpy as np 
from models import shapefree2,BinWerte2
from input_functions import InputWerte
from opti_functions import count_tracers,chi_squared_1,chi_squared_2, chi_squared_3,chi_squared_4,chi_squared_5
import pandas as pd
import os

directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'data') 
ordner1=os.path.join(directory, 'results') 
measFile = os.path.join(ordner, 'VT_VO_HO.xlsx')
carbonageFile=os.path.join(ordner, 'apparent_ages.xlsx')
analyticalFile=os.path.join(ordner1,'analytic.xlsx')

well=63
well_id='VO' #'HO&VO','HO','VO','VT','BW'
values='analytic'

well_name=pd.read_excel(measFile, usecols='A').values
h=pd.read_excel(measFile, usecols='AG').values #whole rows and each value for each well
err_h=pd.read_excel(measFile, usecols='AH').values
NGT=pd.read_excel(measFile, usecols='U').values
err_NGT=pd.read_excel(measFile, usecols='V').values
he=pd.read_excel(measFile, usecols='W').values
err_he=pd.read_excel(measFile, usecols='X').values
ar=pd.read_excel(measFile, usecols='AE').values #not for every well
err_ar=pd.read_excel(measFile, usecols='AF').values
cage=pd.read_excel(carbonageFile, usecols='C').values #empty for some samples
err_cage=pd.read_excel(carbonageFile, usecols='D').values

well_name=well_name[well]
h=h[well]
err_h=err_h[well]
NGT=NGT[well]
err_NGT=err_NGT[well]
he=he[well]
#err_he=err_he[well]
err_he=0.05*he #no error given, 5% of helium as error
ar=ar[well]
err_ar=err_ar[well]
cage=cage[well]
err_cage=err_cage[well]

deep=True
peclet=10 
vogel=True 
temp = 24.0
salinity = 1.0
pressure = 0.955
excess = 2.
 

Tracer='C14'+'Ar39'+'4He'+'3H' #written here should be all the used tracers, all possible tracers are: C14,Ar39,CFC11,CFC12,4He,3H,NGT 
Anzahl_Tracer=count_tracers(Tracer)

t_max=50000
a=np.array([0.,0.,0.])
err_a=np.array([0.,0.,0.])
if well_id=='HO':
    well=well-77
    well_name_2=pd.read_excel(analyticalFile,usecols='A',sheet_name='HO').values[well] 
    a[0]=pd.read_excel(analyticalFile,usecols='M',sheet_name='HO').values[well] ##
    a[1]=pd.read_excel(analyticalFile,usecols='N',sheet_name='HO').values[well]
    a[2]=pd.read_excel(analyticalFile,usecols='O',sheet_name='HO').values[well]
    err_a[0]=pd.read_excel(analyticalFile,usecols='E',sheet_name='HO').values[well]
    err_a[1]=pd.read_excel(analyticalFile,usecols='F',sheet_name='HO').values[well]
    err_a[2]=pd.read_excel(analyticalFile,usecols='G',sheet_name='HO').values[well]

    sd_obsAr=pd.read_excel(analyticalFile,sheet_name='HO',usecols='T').values[well] ## 
    sd_obs3H=pd.read_excel(analyticalFile,sheet_name='HO',usecols='V').values[well]
    sd_obsC14=pd.read_excel(analyticalFile,sheet_name='HO',usecols='U').values[well]
    sd_obs4He=pd.read_excel(analyticalFile,sheet_name='HO',usecols='W').values[well]
    
if well_id=='VO':
    well=well-60
    well_name_2=pd.read_excel(analyticalFile,usecols='A',sheet_name='VO').values[well] 
    a[0]=pd.read_excel(analyticalFile,usecols='M',sheet_name='VO').values[well] ##
    a[1]=pd.read_excel(analyticalFile,usecols='N',sheet_name='VO').values[well]
    a[2]=pd.read_excel(analyticalFile,usecols='O',sheet_name='VO').values[well]
    err_a[0]=pd.read_excel(analyticalFile,usecols='E',sheet_name='VO').values[well]
    err_a[1]=pd.read_excel(analyticalFile,usecols='F',sheet_name='VO').values[well]
    err_a[2]=pd.read_excel(analyticalFile,usecols='G',sheet_name='VO').values[well]

    sd_obsAr=pd.read_excel(analyticalFile,sheet_name='VO',usecols='T').values[well] ## 
    sd_obs3H=pd.read_excel(analyticalFile,sheet_name='VO',usecols='V').values[well]
    sd_obsC14=pd.read_excel(analyticalFile,sheet_name='VO',usecols='U').values[well]
    sd_obs4He=pd.read_excel(analyticalFile,sheet_name='VO',usecols='W').values[well]

#ratios=np.array([0.00328861,0.00311823,0.17792318,8.583e-07,0.73566912,2.82578e-11])
#ratios=np.array([0.006,0.102,0.110,0.074,0.670,0.038])
#err_ratios=np.array([0.004,0.060,0.71,0.056,0.052,0.028])
print(well_name_2,a,err_a)

chi, chi_h,chi_age,chi_he,chi_ar=chi_squared_4(a,t_max,Tracer,Anzahl_Tracer,deep,h,err_h,he,err_he,cage,err_cage,ar,err_ar)
chi2, chi2_h,chi2_age,chi2_he,chi2_ar=chi_squared_5(a,t_max,deep,Tracer, Anzahl_Tracer,h,he,cage,ar,sd_obsAr,sd_obsC14,sd_obs3H,sd_obs4He)
mess_err='Measurement Error'
mo_err='Model Error'
print(chi,chi_age,chi2)
df_results = pd.DataFrame({
    'Well Name': well_name_2.flatten(),
    'Values':values,
    'Mess_err': mess_err,
    'Chi': chi,
    'chi_h': chi_h,
    'chi_age':chi_age,
    'chi_he':chi_he,
    'chi_ar':chi_ar,
    'Model_err': mo_err,
    'Chi2': chi2,
    'chi2_h': chi2_h,
    'chi2_age':chi2_age,
    'chi2_he':chi2_he,
    'chi2_ar':chi2_ar,
    'deep': deep
})

excel_file_path='C:/Users/InesChrista/Bachelorarbeit/Grumpy-master/LPM/results/chi_results.xlsx'
# Write the DataFrame to an Excel file
with pd.ExcelWriter(excel_file_path, mode='a',if_sheet_exists='overlay') as writer:
    df_results.to_excel(writer, index=False, header=False,startrow=writer.sheets["Sheet1"].max_row)

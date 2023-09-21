#chi or chi^2 test

import numpy as np 
from models import shapefree2,BinWerte2
from input_functions import InputWerte
from opti_functions import count_tracers,chi_squared_1,chi_squared_2, chi_squared_3
import pandas as pd
import os

directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'data') 
ordner1=os.path.join(directory, 'results') 
measFile = os.path.join(ordner, 'VT_VO_HO.xlsx')
carbonageFile=os.path.join(ordner, 'apparent_ages.xlsx')
dttdmFile=os.path.join(ordner1, 'DTTDM_results.xlsx')
shapefreeFile=os.path.join(ordner1, 'shapefree.xlsx')

well=65
well_id='VO' #'HO&VO','HO','VO','VT','BW'
values='DTTDM' #'shapefree','DTTDM'

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
 

Tracer='C14'+'Ar39'+'4He'+'3H'+'NGT' #written here should be all the used tracers, all possible tracers are: C14,Ar39,CFC11,CFC12,4He,3H,NGT
Anzahl_Tracer=count_tracers(Tracer)

if well_id=='HO&VO' or well_id=='HO' or well_id=='VO':
    #Bins HO&VO
    bin=[1,2,3,4,5,6] 
    t_grenzen=[0,100,300,1000,10000,25000,50000]
    bin_namen=['<100 years','100-300 years','300-1000 years','1000-10000 years','10000-25000 years','>25000 years']
elif well_id=='BW':
    #Bins BW 
    bin=[1,2,3,4,5] 
    t_grenzen=[0,100,1000,10000,25000,50000]
    bin_namen=['<100 years','100-1000 years','1000-10000 yaers','10000-25000 years','>25000 years']
t_max= np.max(t_grenzen)

rhoC14,rhoAr, rhoC11,rhoC12,rho4He,rho3H,rhoNGT=BinWerte2(InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep),t_grenzen,Tracer,Anzahl_Tracer)

if well_id=='HO':
    well=well-76
    ratios=np.array([0.,0.,0.,0.,0.,0.])
    err_ratios=np.array([0.,0.,0.,0.,0.,0.])
    #well_name_2=pd.read_excel(shapefreeFile, usecols='A',sheet_name='HO').values #shapefreeFile 
    if values=='DTTDM':
        well_name_2=pd.read_excel(dttdmFile, usecols='A',sheet_name='HO').values[well] #dttdmFile
        ratios[0]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='B').values[well]
        ratios[1]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='C').values[well]
        ratios[2]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='D').values[well]
        ratios[3]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='E').values[well]
        ratios[4]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='F').values[well]
        ratios[5]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='G').values[well]
        err_ratios[0]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='H').values[well]
        err_ratios[1]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='I').values[well]
        err_ratios[2]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='J').values[well]
        err_ratios[3]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='K').values[well]
        err_ratios[4]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='L').values[well]
        err_ratios[5]=pd.read_excel(dttdmFile,sheet_name='HO',usecols='M').values[well]
        #print(well_name_2,ratios,err_ratios)
    if values=='shapefree':
        well=well+9+7
        well_name_2=pd.read_excel(shapefreeFile, usecols='A',sheet_name='HO').values[well] #shapefreeFile
        '''
        ratios[0]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='B').values[well]
        ratios[1]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='C').values[well]
        ratios[2]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='D').values[well]
        ratios[3]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='E').values[well]
        ratios[4]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='F').values[well]
        ratios[5]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='G').values[well]
        '''
        ratios[0]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AE').values[well] #the MAP values
        ratios[1]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AF').values[well]
        ratios[2]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AG').values[well]
        ratios[3]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AH').values[well]
        ratios[4]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AI').values[well]
        ratios[5]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AJ').values[well]
        err_ratios[0]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='H').values[well]
        err_ratios[1]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='I').values[well]
        err_ratios[2]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='J').values[well]
        err_ratios[3]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='K').values[well]
        err_ratios[4]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='L').values[well]
        err_ratios[5]=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='M').values[well]
        sd_obsAr=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AP').values[well]
        sd_obsNGT=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AR').values[well]
        sd_obs3H=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AS').values[well]
        sd_obsC14=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AQ').values[well]
        sd_obs4He=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AT').values[well]
        '''
        sd_obsAr=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AO').values[well]
        sd_obsNGT=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AP').values[well]
        sd_obs3H=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AQ').values[well]
        sd_obs4He=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AR').values[well]
        sd_obsC14=0
        '''
        #print(well_name_2,ratios,err_ratios)
if well_id=='VO':
    well=well-59
    ratios=np.array([0.,0.,0.,0.,0.,0.])
    err_ratios=np.array([0.,0.,0.,0.,0.,0.])
    #well_name_2=pd.read_excel(shapefreeFile, usecols='A',sheet_name='HO').values #shapefreeFile 
    if values=='DTTDM':
        well=well+5 #to get the right position in the excel sheet
        well_name_2=pd.read_excel(dttdmFile, usecols='A',sheet_name='VO').values[well] #dttdmFile
        ratios[0]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='B').values[well]
        ratios[1]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='C').values[well]
        ratios[2]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='D').values[well]
        ratios[3]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='E').values[well]
        ratios[4]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='F').values[well]
        ratios[5]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='G').values[well]
        err_ratios[0]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='H').values[well]
        err_ratios[1]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='I').values[well]
        err_ratios[2]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='J').values[well]
        err_ratios[3]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='K').values[well]
        err_ratios[4]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='L').values[well]
        err_ratios[5]=pd.read_excel(dttdmFile,sheet_name='VO',usecols='M').values[well]
        #print(well_name_2,ratios,err_ratios)
    if values=='shapefree':
        well=well+7
        well_name_2=pd.read_excel(shapefreeFile, usecols='A',sheet_name='VO').values[well] #shapefreeFile
        '''
        ratios[0]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='B').values[well]
        ratios[1]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='C').values[well]
        ratios[2]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='D').values[well]
        ratios[3]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='E').values[well]
        ratios[4]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='F').values[well]
        ratios[5]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='G').values[well]
        '''
        ratios[0]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AE').values[well] #the MAP values
        ratios[1]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AF').values[well]
        ratios[2]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AG').values[well]
        ratios[3]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AH').values[well]
        ratios[4]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AI').values[well]
        ratios[5]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AJ').values[well]
        err_ratios[0]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='H').values[well]
        err_ratios[1]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='I').values[well]
        err_ratios[2]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='J').values[well]
        err_ratios[3]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='K').values[well]
        err_ratios[4]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='L').values[well]
        err_ratios[5]=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='M').values[well]
        sd_obsAr=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AP').values[well]
        sd_obsNGT=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AR').values[well]
        sd_obs3H=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AS').values[well]
        sd_obsC14=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AQ').values[well]
        sd_obs4He=pd.read_excel(shapefreeFile,sheet_name='VO',usecols='AT').values[well]
        '''
        sd_obsAr=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AO').values[well]
        sd_obsNGT=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AP').values[well]
        sd_obs3H=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AQ').values[well]
        sd_obs4He=pd.read_excel(shapefreeFile,sheet_name='HO',usecols='AR').values[well]
        sd_obsC14=0
        '''
        #print(well_name_2,ratios,err_ratios)

#ratios=np.array([0.00328861,0.00311823,0.17792318,8.583e-07,0.73566912,2.82578e-11])
#ratios=np.array([0.006,0.102,0.110,0.074,0.670,0.038])
#err_ratios=np.array([0.004,0.060,0.71,0.056,0.052,0.028])
print(well_name_2,ratios,err_ratios)
chi, chi_h,chi_NGT,chi_age,chi_he,chi_ar=chi_squared_1(rhoC14,rhoAr, rhoC11,rhoC12,rho4He,rho3H,rhoNGT, Tracer,t_grenzen, ratios,h,err_h,he,err_he,cage,err_cage,ar,err_ar,NGT,err_NGT)
if values == 'shapefree':
    chi2, chi2_h,chi2_NGT,chi2_age,chi2_he,chi2_ar=chi_squared_3(rhoC14,rhoAr, rhoC11,rhoC12,rho4He,rho3H,rhoNGT, Tracer,t_grenzen, ratios,err_ratios,h,he,cage,ar,NGT,sd_obsAr,sd_obsNGT,sd_obsC14,sd_obs3H,sd_obs4He)
else:
    chi2, chi2_h,chi2_NGT,chi2_age,chi2_he,chi2_ar=chi_squared_2(rhoC14,rhoAr, rhoC11,rhoC12,rho4He,rho3H,rhoNGT, Tracer,t_grenzen, ratios,err_ratios,h,he,cage,ar,NGT)
mess_err='Measurement Error'
mo_err='Model Error'
print(chi,chi_age,chi2)
df_results = pd.DataFrame({
    'Well Name': well_name_2.flatten(),
    'Values':values,
    'Mess_err': mess_err,
    'Chi': chi,
    'chi_h': chi_h,
    'chi_NGT':chi_NGT,
    'chi_age':chi_age,
    'chi_he':chi_he,
    'chi_ar':chi_ar,
    'Model_err': mo_err,
    'Chi2': chi2,
    'chi2_h': chi2_h,
    'chi2_NGT':chi2_NGT,
    'chi2_age':chi2_age,
    'chi2_he':chi2_he,
    'chi2_ar':chi2_ar,
    'deep': deep
})

excel_file_path='C:/Users/InesChrista/Bachelorarbeit/Grumpy-master/LPM/results/chi_results.xlsx'
# Write the DataFrame to an Excel file
with pd.ExcelWriter(excel_file_path, mode='a',if_sheet_exists='overlay') as writer:
    df_results.to_excel(writer, index=False, header=False,startrow=writer.sheets["Sheet1"].max_row)

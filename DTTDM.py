#Discrete Travel Time Distributions Model DTTDM
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import time

start=time.time()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
###Model Setup

well_id='HO' #'HO&VO','HO','VO','VT','BW'
deep=False #this chooses the accumulation rate for 4He: deep=False means shallow 1.5e-11 ccSTP/g, deep=True means deep 4.75e-11ccSTP/g

if well_id=='HO&VO' or well_id=='HO' or well_id=='VO':
    #Bins HO&VO
    bin=[1,2,3,4,5,6] 
    bin_namen=['<100 years','100-300 years','300-1000 years','1000-10000 yaers','10000-25000 years','>25000 years']
    
    #modeled tracer values for each bin
    #3H in TU, 14C age in years, NGT in °C,39Ar in pmAr, 4He shallow in ccSTG/g, 4He deep in ccSTg/g
    bin1_tracer=np.array([7,50,10,88.3,(2.38e-9),(7.50e-10)])
    bin2_tracer=np.array([0.01,200,9.2,60.2,9.5e-9,3e-9])
    bin3_tracer=np.array([0,650,9.3,21.2,3.09e-8,9.75e-9])
    bin4_tracer=np.array([0,5500,9.6,0.3,2.52e-7,7.96e-8])
    bin5_tracer=np.array([0,17500,2.2,0,8.4e-7,2.65e-7])
    bin6_tracer=np.array([0,37500,5,0,1.78e-6,5.63e-7])
    bin_tracer=np.array([bin1_tracer,bin2_tracer,bin3_tracer,bin4_tracer,bin5_tracer,bin6_tracer])

elif well_id=='VT':
    #Bins VT
    bin=[1,2,3,4,5,6]
    bin_namen=['<70 years','70-250 years','250-1000 years','1000-10000 years','10000-25000 years','>25000 years']
    
    #modeled tracer values for each bin
    #3H in TU, 14C age in years, NGT in °C,39Ar in pmAr, 4He shallow in ccSTG/g, 4He deep in ccSTg/g
    bin1_tracer=np.array([7,50,10,88.3,(2.38e-9),(7.50e-10)]) #change to VT
    bin2_tracer=np.array([0.01,200,9.2,60.2,9.5e-9,3e-9]) #change to VT
    bin3_tracer=np.array([0,650,9.3,21.2,3.09e-8,9.75e-9]) #change to VT
    bin4_tracer=np.array([0,5500,9.6,0.3,2.52e-7,7.96e-8])
    bin5_tracer=np.array([0,17500,2.2,0,8.4e-7,2.65e-7])
    bin6_tracer=np.array([0,37500,5,0,1.78e-6,5.63e-7])
    bin_tracer=np.array([bin1_tracer,bin2_tracer,bin3_tracer,bin4_tracer,bin5_tracer,bin6_tracer])

elif well_id=='BW':
    #Bins BW 
    bin=[1,2,3,4,5] 
    bin_namen=['<100 years','100-1000 years','1000-10000 yaers','10000-25000 years','>25000 years'] 

    #modeled tracer values for each bin Broers BW
    #3H in TU, 14C age in years, NGT in °C, 39Ar in pmAr, 4He shallow in ccSTG/g, 4He deep in ccSTg/g
    bin1_tracer=np.array([7,50,10,(2.38e-9),(7.50e-10)])
    bin2_tracer=np.array([0,550,9.3,2.61e-8,8.25e-9])
    bin3_tracer=np.array([0,5500,9.8,2.52e-7,7.96e-8])
    bin4_tracer=np.array([0,17500,2.2,8.4e-7,2.65e-7])
    bin5_tracer=np.array([0,37500,5,1.78e-6,5.63e-7])
    bin_tracer=np.array([bin1_tracer,bin2_tracer,bin3_tracer,bin4_tracer,bin5_tracer])

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Data Input

well=77 #number in the list minus 2

#measured data and errors
directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'data') #weil unterordner
measFile = os.path.join(ordner, 'VT_VO_HO.xlsx')
ageFile=os.path.join(ordner, 'Broers_data.xlsx')
carbonageFile=os.path.join(ordner, 'apparent_ages.xlsx')

h=pd.read_excel(measFile, usecols='AG').values #whole rows and each value for each well
err_h=pd.read_excel(measFile, usecols='AH').values
NGT=pd.read_excel(measFile, usecols='U').values
err_NGT=pd.read_excel(measFile, usecols='V').values
he=pd.read_excel(measFile, usecols='W').values
if well_id=='BW':
    err_he=pd.read_excel(measFile, usecols='X').values
else:
    err_he=he*0.05 #just assumes a 5% error because no error is given
well_name=pd.read_excel(measFile,usecols='A').values
ar=pd.read_excel(measFile, usecols='AE').values #empty for BW samples
err_ar=pd.read_excel(measFile, usecols='AF').values
cage=pd.read_excel(carbonageFile, usecols='C').values #empty for some samples
err_cage=pd.read_excel(carbonageFile, usecols='D').values

well_name=well_name[well]
h=h[well]
err_h=err_h[well]
NGT=NGT[well]
err_NGT=err_NGT[well]
he=he[well]
err_he=err_he[well]
ar=ar[well]
err_ar=err_ar[well]
cage=cage[well]
err_cage=err_cage[well]
#print(well_name,h,err_h,NGT,err_NGT,he,err_he,ar,err_ar,cage,err_cage) #to check the used values

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Chi^2 Calculation (depends on the sample ID) 

chi_r_values=[]
values=np.arange(0, 1.05, 0.05)
valid_combinations = 0  # Count of valid combinations
if well_id=='BW':
    r=[0,0,0,0,0]
    for r0 in values:
        for r1 in values:
            for r2 in values:
                for r3 in values:
                    for r4 in values:
                        if r0 + r1 + r2 + r3 + r4  == 1:
                            r[0] = r0
                            r[1] = r1
                            r[2] = r2
                            r[3] = r3
                            r[4] = r4
                    
                            h_model=r@bin_tracer[:,0]
                            age_model=r@bin_tracer[:,1]
                            NGT_model=r@bin_tracer[:,2]
                            he1_model=r@bin_tracer[:,3]
                            he2_model=r@bin_tracer[:,4]

                            #chi square approach: chi^2=(model-measure)^2/(error)^2 and chi^2(sample)=chi^2(tracer1)+chi^2(tracer2)+...
                            chi_h=(h_model-h)**2/(err_h)**2
                            chi_NGT=(NGT_model-NGT)**2/(err_NGT)**2
                            chi_age=(age_model-cage)**2/(err_cage)**2
                            if deep==True:
                                chi_he=(he2_model-he)**2/(err_he)**2
                            else:
                                chi_he=(he1_model-he)**2/(err_he)**2
                            chi=chi_h+chi_NGT+chi_he+chi_age
                            valid_combinations += 1
                            chi_r_values.append((chi, r.copy()))
else:
    r=[0,0,0,0,0,0]
    for r0 in values:
        for r1 in values:
            for r2 in values:
                for r3 in values:
                    for r4 in values:
                        for r5 in values:
                            if r0 + r1 + r2 + r3 + r4 + r5  == 1:
                                r[0] = r0
                                r[1] = r1
                                r[2] = r2
                                r[3] = r3
                                r[4] = r4
                                r[5] = r5
                        
                                h_model=r@bin_tracer[:,0]
                                age_model=r@bin_tracer[:,1]
                                NGT_model=r@bin_tracer[:,2]
                                ar_model=r@bin_tracer[:,3]
                                he1_model=r@bin_tracer[:,4]
                                he2_model=r@bin_tracer[:,5]

                                #chi square approach: chi^2=(model-measure)^2/(error)^2 and chi^2(sample)=chi^2(tracer1)+chi^2(tracer2)+...
                                chi_h=(h_model-h)**2/(err_h)**2
                                chi_NGT=(NGT_model-NGT)**2/(err_NGT)**2
                                chi_age=(age_model-cage)**2/(err_cage)**2
                                chi_ar=(ar_model-ar)**2/(err_ar)**2
                                if deep==True:
                                    chi_he=(he2_model-he)**2/(err_he)**2
                                else:
                                    chi_he=(he1_model-he)**2/(err_he)**2
                                chi=chi_h+chi_NGT+chi_ar+chi_he+chi_age
                                valid_combinations += 1
                                chi_r_values.append((chi, r.copy()))

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Optimization

# Sort the list of chi-square and r values based on chi-square in ascending order
chi_r_values = sorted(chi_r_values, key=lambda x: x[0])

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Results 

# Display the 50 smallest chi-square values and corresponding r values
print("50 smallest chi-square values and corresponding r values:")
r_best=[]
for i in range(50):
    chi, r_values = chi_r_values[i]
    #print("Chi-square:", chi)
    #print("r values:", r_values)
    r_best.append(r_values)

    h_model = r_values @ bin_tracer[:, 0]
    age_model = r_values @ bin_tracer[:, 1]
    NGT_model = r_values @ bin_tracer[:, 2]
    if well_id=='BW':
        he1_model = r_values @ bin_tracer[:, 3]
        he2_model = r_values @ bin_tracer[:, 4]
    else:
        ar_model = r_values @ bin_tracer[:, 3]
        he1_model = r_values @ bin_tracer[:, 4]
        he2_model = r_values @ bin_tracer[:, 5]

    chi_h = (h_model - h) ** 2 / (err_h) ** 2
    chi_NGT = (NGT_model - NGT) ** 2 / (err_NGT) ** 2
    chi_age=(age_model-cage) ** 2 / (err_cage) ** 2
    if deep == True:
        chi_he = (he2_model - he) ** 2 / (err_he) ** 2
    else:
        chi_he = (he1_model - he) ** 2 / (err_he) ** 2

    print(f"Chi-square values for Combination {i + 1}:")
    print("H:", chi_h)
    print("NGT:", chi_NGT)
    
    print("He:", chi_he)
    print("C14 age:", chi_age)
    if well_id=='HO&VO' or well_id=='HO' or well_id=='VO' or well_id=='VT':
        chi_ar = (ar_model - ar) ** 2 / (err_ar) ** 2
        print("Ar:", chi_ar)
    print("Total Chi-square:", chi)

print("Number of valid combinations:", valid_combinations)
#print(r_best)

r_mean = np.mean(r_best, axis=0)
# Calculate the standard deviation of the r arrays
r_std = np.std(r_best, axis=0)

end=time.time()

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
### Output

# Display the mean and standard deviation of the r arrays
print("Well ",(well_name))
print("Mean of r arrays:", np.round(r_mean,4)*100)
print("Standard deviation of r arrays:", np.round(r_std,4)*100)
print('time DTTDM',(end-start))

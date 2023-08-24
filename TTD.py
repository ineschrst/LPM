#Die jeweiligen Transit Time Distribution der Modelle zusammengefasst
#alle TTDs auch in Models zu finden 
#plots
#von ines geschrieben und getestet

import numpy as np
import matplotlib.pyplot as plt 


#shapefree
#plot der Bins der einzelnen Tracer also im Prinzip Plot der Ratios
def shapefreeTTD(a,bins): 
    #bins=np.arange(numberofbins) #Bin 1,2,3,4,5... t_grenzen müssten ja an den Enden der Bins stehen->geht wahrscheinlich auch
    plt.bar(bins,a,width=0.5)
    plt.xlabel("Age Bins")
    plt.ylabel("Ratio of each age bin") #how much water is how old
    plt.title("Contributions of the age bins to the water mixture")
    plt.show()
    return () 



#inverse Gaussian
#peclet zahl könnte man auch als Variable festlegen
def inverseGaussianTTD(a,t_max): #ttd plotten 
    peclet=10
    zeitIG = np.arange(t_max)
    #wertIG1 = np.zeros(len(zeitIG)) braucht es das?

    q1 = np.sqrt(peclet * a / (4 * np.pi)) #Vorfaktor
    wertIG = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - a) ** 2) / (4 * a * zeitIG)) #inverse gaussian 

    #wertIG1[0] = wertIG1[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertIG, copy=True)

    plt.plot(zeitIG,wertIG)
    plt.xlabel('Time in years')
    plt.ylabel('Distribution')
    plt.title('Inverse Gaussian TTD')
    plt.show()
    
    return
#RuntimeWarning: divide by zero encountered in power/divide/multiply
# wertIG = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - a) ** 2) / (4 * a * zeitIG)) #inverse gaussian


def inverseGaussianMixTTD(a,t_max): #ttd des mixes plotten
    peclet=10
    tau1=a[0]
    tau2=a[1]
    ratio=a[2]

    zeitIG = np.arange(t_max)
    #wertIG1 = np.zeros(len(zeitIG)) braucht es das?

    q1 = np.sqrt(peclet * tau1 / (4 * np.pi)) #Vorfaktor
    q2 = np.sqrt(peclet * tau2 / (4 * np.pi)) #Vorfaktor
    wertIG1 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) / (4 * tau1 * zeitIG)) #inverse gaussian 
    wertIG2= q2 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau2) ** 2) / (4 * tau2 * zeitIG))

    #wertIG1[0] = wertIG1[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertIG1, copy=True)
    np.nan_to_num(wertIG2,copy=True)

    wertIG=ratio*wertIG1+(1-ratio)*wertIG2

    plt.plot(zeitIG,wertIG)
    plt.xlabel('Time in years')
    plt.ylabel('Distribution')
    plt.title('Inverse Gaussian Mix TTD')
    plt.show()
    
    return
#RuntimeWarning: divide by zero encountered in power/divide/multiply  
# wertIG1/2 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) / (4 * tau1 * zeitIG)) #inverse gaussian


#Gamma 
def gammaTTD(a,shape,G,t_max): #a ist wieder tau
    beta=a/shape
    
    zeit=np.arange(t_max)

    wertgamma=zeit ** (shape - 1.) / (beta ** (shape) * G) * np.exp(-zeit/beta)
    np.nan_to_num(wertgamma, copy=True)

    plt.plot(zeit,wertgamma)
    plt.xlabel('Time in years')
    plt.ylabel('Distribution')
    plt.title('Gamma TTD')
    plt.show()

    return


#exponential
def exponentialTTD(a,t_max):
    zeit=np.arange(t_max)
    wertex=1/a*np.exp(-zeit/a)
    np.nan_to_num(wertex,copy=True)

    plt.plot(zeit,wertex)
    plt.xlabel('Time in years')
    plt.ylabel('Distribution')
    plt.title('Exponential TTD')
    plt.show()

    return 
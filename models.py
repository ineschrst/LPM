###pistonflow
#assumes no mixing or dispersion
#used when dispersion is low, flow velocity high or flow path short (Tracer LPM Page 4)
#no distribution, but rather water from one age, the mean age
#kopiert von Viola
import numpy as np
import pymc.math as pmm

def pistonMix(a, inputWerte, t_max): #a sind die Mean Ages der zwei Wassermassen und die Ratio; inputWerte sind die Atmosphärischen Werte zu der Zeit; t_max ist das maximale Alter

    tau1 = int(a[0])
    tau2 = int(a[1])
    ratio = a[2]

    if tau1 > t_max or tau2 > t_max:
        print("t_max ist zu klein gewählt! MRT außerhalb der Grenzen!")
    c_calc1 = np.array([0., 0., 0., 0.]) #in diesem Programm gibt es vier Tracer von denen die Konzentration kalkuliert wird
    c_calc2 = np.array([0., 0., 0., 0.])

    N = np.arange(4)
    for i in N:
        c_calc1[i] = inputWerte[tau1, i] #in den Inputwerten wird die Konzentration zur Zeit tau1 nachgeschaut und als kalkulierte Konzentration angegeben 
        c_calc2[i] = inputWerte[tau2, i]
    c_calc = ratio * c_calc1 + (1 - ratio) * c_calc2 #da hier zwei Wassermassen gemixt werden, wird die Konzentration der einzelnen miteinander verrechnet um die Gesamtkonzentration der jeweiligen Tracer zu erhalten
    
    return c_calc

import numpy as np

def piston(a, inputWerte, t_max): #eine Wassermasse statt Mix aus zwei, stattdessen ist auch eine Ratio von 1 oder 0 möglich bzw annähernd so -> prinzipiell gleiches Ergebnis

    tau = int(a[0])

    if tau > t_max:
        print("t_max ist zu klein gewählt! MRT außerhalb der Grenzen!")
    c_calc = np.array([0., 0., 0., 0.])

    N = np.arange(4)
    for i in N:
        c_calc[i] = inputWerte[tau, i] 

    return c_calc





###shapefree 
#model the Transit Time Distribution in bins rather than using an analytical Model
#allows to also model groundwater with multiple mean ages -> mixed 
#carefull to not overfit (maybe not called overfitting) -> number of tracer measurements used should be at least k+1 where k is the number of age bins (at least it should be somehow restricted)
#von Viola kopiert 
import os
import numpy as np
import matplotlib.pyplot as plt
#ausgeblendet damit die anderen Funktionen laufen -> zum testen von meinen selbstbeschriebenen Funktionen

from grumpy.inputfunktionen import inputC14,inputAr39,inputCFC11,inputCFC12 #alle inputfunktionen

directory = os.path.dirname(os.path.abspath(__file__))
outputPath = os.path.join(directory, 'Output')
inputPath = os.path.join(directory, 'Input')
inputFile = os.path.join(inputPath, 'Input_functions.xlsx')
measFile = os.path.join(inputPath, 'Datierung_Oman_05_2018.xlsx')


def ungewichteteBinWerte_variabel (t_grenzen, temp, salinity, pressure, excess, vogel, t_max): #len(t_grenzen)/anzahl bins sollte über die Anzahl an Tarcern beschränkt sein
    # in t_Grenzen werden die Bingrenzen festgelegt. Z.B. ergibt [0, 100, 1000, 20000] 3 Bins
    #print("t_max=", t_max)
    zeitC14, wertC14 = inputC14(vogel, t_max) #nur die Inputfunktionen für die es auch Messdaten gibt, sonst weglassen
    zeitAr39, wertAr39 = inputAr39(t_max) #sind die Zeitwerte überhaupt relevant?? 
    zeitCFC11, wertCFC11 = inputCFC11(temp, salinity, pressure, excess, t_max) #Wert der Inputfunktion bei Zeit
    zeitCFC12, wertCFC12 = inputCFC12(temp, salinity, pressure, excess, t_max)

    anzahl_bins = len(t_grenzen) - 1
    #print("Anzahl Bins: ", anzahl_bins)

    rho = np.zeros((4, anzahl_bins))  # Matrix mit Konzentrationen fuer Bins und Tracer (Dimensionen: 4 x anzahlBins)
    n = 0                             # Dimensionen Matrix Anzahl Tracer x anzahl bins

    for i in np.arange(anzahl_bins):
        # print("ungewichteteBinWerte_variabel, arbeite an Bin ", i)
        # print("Zwischen diesen Zeiten: ", t_Grenzen[i], "und ", t_Grenzen[i + 1])
        while (n >= t_grenzen[i]) and (n < t_grenzen[i+1]):
            c = wertC14[n] / (t_grenzen[i+1] - t_grenzen[i]) #über die Zeit normalisiert 
            d = wertAr39[n] / (t_grenzen[i+1] - t_grenzen[i]) #zu zeit 1,2,3,4 ... wird jeweils die konzentration aus dem Input genommen
            e = wertCFC11[n] / (t_grenzen[i+1] - t_grenzen[i])
            f = wertCFC12[n] / (t_grenzen[i+1] - t_grenzen[i])

            rho[0, i] = rho[0, i] + c  # C14  #in jedem Bin wird die Konzentration von den n verschiedenen Zeiten innerhalb aufaddiert
            rho[1, i] = rho[1, i] + d  # Ar39 #Mittelwert Berechnung innerhalb der Bins
            rho[2, i] = rho[2, i] + e  # CFC11
            rho[3, i] = rho[3, i] + f  # CFC12
            n += 1


    return rho


def nonparametric_variabel(a, rho):  #statt A rho, da bei dem Bin werte variable Modell A nicht verwendet wurde
    # len(a) = Anzahl Bins = Anzahl freie Parameter + 1
    # Shape A:
    # print("A hat folgende Shape: ", A.shape)
    # print("a hat folgende Shape: ", a.shape)
    c_calc = np.dot(rho, a)  # Skalarprodukt zw. der Matrix ungewichteteBinWerte_variabel und dem Gewichtungsvektor a #a sind die Ratios der einzelnen Bins c_calc=c_calcbin1*ratio1+...
    # c_calc[1] = np.around(c_calc[1], 0)  # da Ar-39 einer Poissonverteilung folgt, muss es ein integer sein. VERALTET
    return c_calc  # Vektor der Form 4x1 #Vektor AnzahlderTarcerx1






###inverse Gaussian or DispersionsModell 
#need two parameters -> mean age and dispersions Pararmeter (inverse of Peclet Number)
#describes a flow system with (limited) mixing due to dispersion 
#Scale analysis constrainst Pecelet Number to about 10 -> 1/10=0.1 

import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import time


def plot_inverseGaussianMix(a, t_max, info):  # der Plot sieht richtig kacke aus, wird wsh keine Verwendung finden #inverse Gaussian mit zwei Wassermassen
    peclet = 10 #festgelegt 
    tau1 = a[0] 
    tau2 = a[1]
    ratio = a[2]

    # print(inputWerte[:,1])
    zeitIG = np.arange(t_max)
    wertIG1 = np.zeros(len(zeitIG))  # leeres Array für Altersverteilung für Wassermasse 1
    wertIG2 = np.zeros(len(zeitIG))  # leeres Array für Altersverteilung für Wassermasse 2

    q1 = np.sqrt(peclet * tau1 / (4 * np.pi))  # hängt nur von tau ab, nicht von t -> ziehe es aus der for-schleife
    q2 = np.sqrt(peclet * tau2 / (4 * np.pi))

    for n in (1, zeitIG):
        wertIG1[n] = ratio * q1 * n ** (-3 / 2) * np.exp((-peclet * (n - tau1) ** 2) / (4 * tau1 * n)) #statt for Schleife geht auch zeitIG statt n einzusetzten siehe InverseGausian für eine Wassermasse
        wertIG2[n] = (1 - ratio) * q2 * n ** (-3 / 2) * np.exp((-peclet * (n - tau2) ** 2) / (4 * tau2 * n))

    wertIG1[0] = wertIG1[1] #das erste Element soll nicht null sein -> in der efunktion steht n unter dem Bruchstrich also kann man keine Null einsetzten
    wertIG2[0] = wertIG2[1]

    np.nan_to_num(wertIG1, copy=True) 
    np.nan_to_num(wertIG2, copy=True)

    intIG1 = np.sum(wertIG1) #aufsummiert sollte die Wahrscheinlichkeitsverteilung 1 geben
    intIG2 = np.sum(wertIG2)

    #plot rausgelassen

    if 0.98 < (intIG1 + intIG2) < 1.02: #eventuell zu strenge Grenzen 
        print("Achtung, in InverseGaussianMix wird eine Grenze erreicht! tau1 = ", tau1, "tau2 = ", tau2)
        print("Integral 1 =", intIG1, "Integral 2 = ", intIG2, ", Summe: ", intIG1+intIG2)
#ohne plot sinnlos


def inverseGaussianMix(a,b,c, inputWerte, t_max, peclet,Anzahl_Tracer,Tracer): #man könnte es hier auch rauslassen
    # das normale Dispersion model, für zwei Wassermassen
    # im Paper als DMmix bezeichnet
    tau1 = a #warum +.01?? und float nicht notwendig
    tau2 =b #wenn das ist weil bei zeit 0.1 addiert wird müsste es auch 0.1 sein statt .01
    ratio = c
    peclet = 10 #festgelegt, statt zwei peclets

    zeitIG = np.arange(t_max, dtype=float)
    zeitIG[0] = 0.1 #vielleicht das nicht mit der Zeit 0 gearbeitet wird??
    wertIG1 = np.zeros(len(zeitIG))
    wertIG2 = np.zeros(len(zeitIG))
    # t1 = time.perf_counter()

    # hängt nur von tau & peclet ab, nicht von t -> berechne es vor Vektormultiplikation
    q1 = np.sqrt(peclet * tau1/(4 * np.pi))
    q2 = np.sqrt(peclet * tau2/(4 * np.pi))
    # t2 = time.perf_counter()
    wertIG1 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) /
                                                                 (4 * tau1 * zeitIG))
    wertIG2 = q2 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau2) ** 2) /
                                                                 (4 * tau2 * zeitIG))
    wertIG1[0]=wertIG1[1]
    wertIG2[0]=wertIG2[1]

    intIG1 = np.sum(wertIG1)
    intIG2 = np.sum(wertIG2)

    c_calc1=np.zeros(Anzahl_Tracer)  #np.zeros?? dann könnte man eine weitere Variable für die Anzahl an Tracern einfügen
    c_calc2=np.zeros(Anzahl_Tracer)
    c_calc=np.zeros(Anzahl_Tracer)
    k=0
    if 'C14' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'Ar39' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'CFC11' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'CFC12' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1

    #print("c1", c_calc1, "c2", c_calc2) #fehlersuche ines
    return c_calc
def inverseGaussianMixvary(a,b,c, inputWerte, t_max, peclet,Anzahl_Tracer,Tracer): #vary ratio
    tau1 = a #warum +.01?? und float nicht notwendig
    tau2 = b #wenn das ist weil bei zeit 0.1 addiert wird müsste es auch 0.1 sein statt .01
    ratio = c
    peclet = 10 #festgelegt, statt zwei peclets

    zeitIG = np.arange(t_max, dtype=float)
    zeitIG[0] = 0.1 #vielleicht das nicht mit der Zeit 0 gearbeitet wird??
    wertIG1 = np.zeros(len(zeitIG))
    wertIG2 = np.zeros(len(zeitIG))
    # t1 = time.perf_counter()

    # hängt nur von tau & peclet ab, nicht von t -> berechne es vor Vektormultiplikation
    q1 = np.sqrt(peclet * tau1/(4 * np.pi))
    q2 = np.sqrt(peclet * tau2/(4 * np.pi))
    # t2 = time.perf_counter()
    wertIG1 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) /
                                                                 (4 * tau1 * zeitIG))
    wertIG2 = q2 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau2) ** 2) /
                                                                 (4 * tau2 * zeitIG))
    wertIG1[0]=wertIG1[1]
    wertIG2[0]=wertIG2[1]

    c_calc1=np.zeros(Anzahl_Tracer)  #np.zeros?? dann könnte man eine weitere Variable für die Anzahl an Tracern einfügen
    c_calc2=np.zeros(Anzahl_Tracer)
    c_calc=np.zeros((Anzahl_Tracer,len(c)))
    k=0
    if 'C14' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'Ar39' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'CFC11' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'CFC12' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    #print("c1", c_calc1, "c2", c_calc2) #fehlersuche ines
    return c_calc
def inverseGaussianMixvary1(a,b,c, inputWerte, t_max, peclet,Anzahl_Tracer,Tracer): #vary age1
    tau1 = a #warum +.01?? und float nicht notwendig
    tau2 = b #wenn das ist weil bei zeit 0.1 addiert wird müsste es auch 0.1 sein statt .01
    ratio = c
    peclet = 10 #festgelegt, statt zwei peclets

    zeitIG = np.arange(t_max, dtype=float)
    zeitIG[0] = 0.1 #vielleicht das nicht mit der Zeit 0 gearbeitet wird??
    wertIG1 = np.zeros((len(a),len(zeitIG)))
    wertIG2 = np.zeros(len(zeitIG))
    q1=np.zeros(len(a))
    for i in np.arange(len(a)):
        q1[i] = np.sqrt(peclet * tau1[i]/(4 * np.pi))
        wertIG1[i] = q1[i] * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1[i]) ** 2) /(4 * tau1[i] * zeitIG))

    q2 = np.sqrt(peclet * tau2/(4 * np.pi))

    wertIG2 = q2 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau2) ** 2) /
                                                                 (4 * tau2 * zeitIG))
    wertIG1[:,0]=wertIG1[:,1]
    wertIG2[0]=wertIG2[1]

    c_calc1=np.zeros((Anzahl_Tracer,len(a)))  #np.zeros?? dann könnte man eine weitere Variable für die Anzahl an Tracern einfügen
    c_calc2=np.zeros(Anzahl_Tracer)
    c_calc=np.zeros((Anzahl_Tracer,len(a)))
    k=0
    if 'C14' in Tracer:
        for n in np.arange(len(a)):
            c_calc1[k,n]=np.tensordot(inputWerte[:,k],wertIG1[n,:],axes=([0],[0]))
            c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k,n]+(1-ratio)*c_calc2[k]
        k=k+1
    if 'Ar39' in Tracer:
        for n in np.arange(len(a)):
            c_calc1[k,n]=np.tensordot(inputWerte[:,k],wertIG1[n,:],axes=([0],[0]))
            c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k,n]+(1-ratio)*c_calc2[k]
        k=k+1
    if 'CFC11' in Tracer:
        for n in np.arange(len(a)):
            c_calc1[k,n]=np.tensordot(inputWerte[:,k],wertIG1[n,:],axes=([0],[0]))
            c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k,n]+(1-ratio)*c_calc2[k]
        k=k+1
    if 'CFC12' in Tracer:
        for n in np.arange(len(a)):
            c_calc1[k,n]=np.tensordot(inputWerte[:,k],wertIG1[n,:],axes=([0],[0]))
            c_calc2[k]=np.tensordot(inputWerte[:,k],wertIG2,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k,n]+(1-ratio)*c_calc2[k]
        k=k+1

    return c_calc
def inverseGaussianMixvary2(a,b,c, inputWerte, t_max, peclet,Anzahl_Tracer,Tracer): #vary age2
    tau1 = a #warum +.01?? und float nicht notwendig
    tau2 = b #wenn das ist weil bei zeit 0.1 addiert wird müsste es auch 0.1 sein statt .01
    ratio = c
    peclet = 10 #festgelegt, statt zwei peclets

    zeitIG = np.arange(t_max, dtype=float)
    zeitIG[0] = 0.1 #vielleicht das nicht mit der Zeit 0 gearbeitet wird??
    wertIG2 = np.zeros((len(b),len(zeitIG)))
    wertIG1 = np.zeros(len(zeitIG))
    q2=np.zeros(len(b))
    for i in np.arange(len(b)):
        q2[i] = np.sqrt(peclet * tau2[i]/(4 * np.pi))
        wertIG2[i] = q2[i] * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau2[i]) ** 2) /(4 * tau2[i] * zeitIG))

    q1 = np.sqrt(peclet * tau1/(4 * np.pi))

    wertIG1 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) /
                                                                 (4 * tau1 * zeitIG))
    wertIG2[:,0]=wertIG2[:,1]
    wertIG1[0]=wertIG1[1]

    c_calc2=np.zeros((Anzahl_Tracer,len(b)))  #np.zeros?? dann könnte man eine weitere Variable für die Anzahl an Tracern einfügen
    c_calc1=np.zeros(Anzahl_Tracer)
    c_calc=np.zeros((Anzahl_Tracer,len(b)))
    k=0
    
    if 'C14' in Tracer:
        for n in np.arange(len(b)):
            c_calc2[k,n]=np.tensordot(inputWerte[:,k],wertIG2[n,:],axes=([0],[0]))
            c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k,n]
        k=k+1
    if 'Ar39' in Tracer:
        for n in np.arange(len(b)):
            c_calc2[k,n]=np.tensordot(inputWerte[:,k],wertIG2[n,:],axes=([0],[0]))
            c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k,n]
        k=k+1
    if 'CFC11' in Tracer:
        for n in np.arange(len(b)):
            c_calc2[k,n]=np.tensordot(inputWerte[:,k],wertIG2[n,:],axes=([0],[0]))
            c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k,n]
        k=k+1
    if 'CFC12' in Tracer:
        for n in np.arange(len(b)):
            c_calc2[k,n]=np.tensordot(inputWerte[:,k],wertIG2[n,:],axes=([0],[0]))
            c_calc1[k]=np.tensordot(inputWerte[:,k],wertIG1,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k,n]
        k=k+1

    return c_calc



def igpistonMix(a, inputWerte, t_max):  # junge Komponente: Inverse Gaussian, alte Komponente: Piston-Flow
    peclet = 10
    tau1 = a[0]  # inverse Gaussian
    tau2 = int(a[1])  # piston-flow
    ratio = a[2]

    # print(inputWerte[:,1])
    zeitIG = np.arange(t_max)
    # wertIG1 = np.zeros(len(zeitIG)) # Altersverteilung der jungen (IG) Komponente

    q1 = np.sqrt(peclet * tau1 / (4 * np.pi))
    wertIG1 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) / (4 * tau1 * zeitIG))
    # for n in (1, zeitIG):
    #    wertIG1[n] = q1 * n ** (-3 / 2) * np.exp((-peclet * (n - tau1) ** 2) / (4 * tau1 * n))
    wertIG1[0] = wertIG1[1]
    np.nan_to_num(wertIG1, copy=True) #ist die nan nicht schon durch das ersetzten des ersten Wertes erledigt??

    # plt.plot(zeitIG, wertIG1)
    # plt.xlim(0, tau2*2)
    # figname = 'C:\\Users\\vraedle\\Masterarbeit\\Paper\\TTDs\\TTD_IG_' + str(tau1) + '_' + str(tau2) + '.png'
    # plt.savefig(figname, bbox_inches='tight', dpi=500)

    intIG1 = np.sum(wertIG1)
    if intIG1 < 0.88 :
        print("Achtung, in InverseGaussianMix wird eine Grenze erreicht! tau1 = ", tau1)
        print("Integral 1 =", intIG1)

    c_calc1 = np.array([0., 0., 0., 0.])
    c_calc2 = np.array([0., 0., 0., 0.])

    N = np.arange(4)
    for i in N:
        c_calc1[i] = np.dot(inputWerte[:, i], wertIG1)  # Skalarprodukt
        c_calc2[i] = inputWerte[tau2, i]  # Piston-Flow: Altersverteilung entspricht zu 100% der Konzentration aus Jahr tau2

    c_calc = ratio * c_calc1 + (1 - ratio) * c_calc2

    return c_calc


def pistonIGMix(a, inputWerte, t_max): # junge Komp: piston-flow, alte Komp: Inverse Gaussian
    peclet = 10

    tau1 = int(a[0])  # piston-flow
    tau2 = a[1]  # Inverse Gaussian
    ratio = a[2]

    zeitIG = np.arange(t_max)
    # wertIG2 = np.zeros(len(zeitIG))

    q2 = np.sqrt(peclet * tau2 / (4 * np.pi))
    wertIG2 = q2 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau2) ** 2) / (4 * tau2 * zeitIG))
    # for n in (1, zeitIG):
    #    wertIG2[n] = q2 * n ** (-3 / 2) * np.exp((-peclet * (n - tau2) ** 2) / (4 * tau2 * n))
    wertIG2[0] = wertIG2[1]
    np.nan_to_num(wertIG2, copy=True)

    intIG2 = np.sum(wertIG2)
    if intIG2 < 0.88 :
        print("Achtung, in InverseGaussianMix wird eine Grenze erreicht! tau2 = ", tau2)
        print("Integral 2 =", intIG2)

    c_calc1 = np.array([0., 0., 0., 0.])
    c_calc2 = np.array([0., 0., 0., 0.])

    N = np.arange(4) #iteriert über Tracer
    for i in N:
        c_calc1[i] = inputWerte[tau1, i]  # piston-flow: 100% Gewichtung von Jahr tau1
        c_calc2[i] = np.dot(inputWerte[:, i], wertIG2)  # Skalarprodukt

    c_calc = ratio * c_calc1 + (1 - ratio) * c_calc2

    return c_calc
#das die junge Komponente DM ist und die alte piston ist erscheint sehr unwahrscheinlich da das piston modell
#  eher bei kurzen flow paths(also auch Aufenthaltszeiten meistens) angewandt werden kann

def inverseGaussian(a, inputWerte, t_max):  # für nur eine Wassermasse. Bisher nicht in Verwendung 
    peclet = 10 #aus der Scale Analysis festgelegt
    tau1 = a[0]

    # print(inputWerte[:,1])
    zeitIG = np.arange(t_max)
    wertIG1 = np.zeros(len(zeitIG))

    q1 = np.sqrt(peclet * tau1 / (4 * np.pi)) #Vorfaktor
    wertIG1 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) / (4 * tau1 * zeitIG)) #inverse gaussian 

    # for n in (1, zeitIG):
    #    wertIG1[n] = q1 * n ** (-3 / 2) * np.exp((-peclet * (n - tau1) ** 2) / (4 * tau1 * n)) #not needed loop

    wertIG1[0] = wertIG1[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertIG1, copy=True)
    

    intIG1 = np.sum(wertIG1) #aufsummierte Distribution sollte eins ergeben

    if intIG1 < 0.88: #tau ist so nah an tmax das ein Teil der Verteilung nicht mehr dargestellt wird deswegen ist der auf integrierte Wert kleiner als 1
        print("Achtung, in InverseGaussianMix wird eine Grenze erreicht! tau1 = ", tau1)
        print("Integral 1 =", intIG1)

    c_calc1 = np.array([0., 0., 0., 0.]) #für mehr Tracer müssen dann auch mehr kalkulierte Konzentrationen rauskommen 

    N = np.arange(4)
    for i in N:
        c_calc1[i] = np.dot(inputWerte[:, i], wertIG1) #die Verteilung wird mit den Konzentrationswerten verbunden mithilfe eines Skalarprdouktes 

    return c_calc1 

#ines getestet 
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
    plt.show()
    
    return



###gamma model
#gamma distribution gibt es scheinbar unter stats.gamma.pdf (import scipy.stats as stats)
#Gamma mixture model is more robust in estimating pollution with multiple sources -> maybe means multiple tracers would also be good??
#amount of rainfall accumulated in a resoirvar can be modelled with gamma
#super komplex -> noch verstehen

#von viola kopiert
import numpy as np
import concurrent.futures

import time
# import xlsxwriter  #wird nicht genutzt, genau wie datetime, quad und getInput 
from datetime import datetime
from scipy.integrate import quad
#from inputfunktionen import getInput
import matplotlib.pyplot as plt

def gammaDiffShapesMix(a, shape, G, inputWerte, t_max):  #shape sind die Gammaparameter, G ist die Gammafunktion(shapes)=(shapes-1)!

    inputWerte=np.transpose(inputWerte)

    beta1 = a[0]/shape[0] #statt beta anzugeben ist es einfacher wie immer tau anzugeben und beta auszurechnen -> beta=tau/shape
    beta2 = a[1]/shape[1] #wenn tau geraten werden kann man auch beta raten allerdings wenn es schon grobe werte gibt ist tau besser -> wäre auch konsistent mit den anderen angaben von a
    ratio = a[2]
    if len(shape) == 1: #je nach dem ob ein oder zwei Shapes angegeben werden -> k (oder p je nach Quelle)
        shape1, shape2 = shape
        G1, G2 = G
        print("hier wurden keine zwei Shapes übergeben, sondern nur eine! G = ", shape)

    if len(shape) == 2:
        shape1 = shape[0]
        shape2 = shape[1]
        G1 = G[0]
        G2 = G[1]
        #print("Shapes sind ", shape1, shape2, "mit G ", G1, G2, "bei t max ", t_max, "mit betas ", beta1, beta2)

 #G könnte auch aus k berechnet werden -> statt anzugebeben 

    zeit = np.arange(t_max)
    wertGamma1=np.zeros(len(zeit))  #wegen (20,) und (20,1) Fehler ines
    wertGamma2=np.zeros(len(zeit))

    wertGamma1 = zeit ** (shape1 - 1.) / (beta1 ** (shape1) * G1) * np.exp(-zeit/beta1)
    wertGamma2 = zeit ** (shape2 - 1.) / (beta2 ** (shape2) * G2) * np.exp(-zeit/beta2)

    wertGamma1[0] = 0.1 ** (shape1 - 1.) / (G1 * beta1 ** (shape1)) * np.exp(-0.1 / beta1)  # zeit=0 geht mathematisch nicht, nehme stattdessen zeit=0.1
    wertGamma2[0] = 0.1 ** (shape2 - 1.) / (G2 * beta2 ** (shape2)) * np.exp(-0.1 / beta2)

    #np.nan_to_num(wertGamma1, copy=True)  #statt mit zeit=0.1 zu beginnen kann man auch den nan in eine 0 umwandeln lassen, allerdings ist zu zeit 0 die Verteilung nicht umbedingt 0 -> nur bei bestimmten Vorraussetzungen
    #np.nan_to_num(wertGamma2, copy=True)
    #print(zeit.shape)
    #print(wertGamma1.shape)
    #wertGamma1=wertGamma1.reshape() #ines
    #wertGamma2=wertGamma2.reshape((len(zeit)),(1)) #ines
    #print(wertGamma1.shape)

    intGamma1 = np.sum(wertGamma1)
    intGamma2 = np.sum(wertGamma2)

    if (intGamma1 < 0.9 and beta1*shape1 > 10) or intGamma2 < 0.88 or intGamma1 > 1.1 or intGamma2 > 1.1: #warum darf tau1 nicht größer als 10 sein?
        print("Achtung, in gammaMix ist die Unter-/Obergrenze (t_max) erreicht! tau1 = ", beta1*shape1, "tau2 = ", beta2*shape2)
        print("Integral 1 =", intGamma1, "Integral 2 = ", intGamma2)

    c_calc1 = np.array([0., 0., 0., 0.])
    c_calc2 = np.array([0., 0., 0., 0.])

    N = np.arange(4)
    for i in N:
        c_calc1[i] = np.dot(inputWerte[i,:], wertGamma1) #geht mit meiner Matrix für die Inputwerte
        c_calc2[i] = np.dot(inputWerte[i,:], wertGamma2) #die Inputwerte transponieren funktuniert nicht, aber wenn [i,:] verwendet wird geht es scheinbar

    #for i in N:
        #c_calc1[i] = np.dot(inputWerte[:, i], wertGamma1) (ursprünglich)
        #c_calc2[i] = np.dot(inputWerte[:, i], wertGamma2)

    c_calc = ratio * c_calc1 + (1 - ratio) * c_calc2

    return c_calc  #wie immer 

#ines (getestet)
def gammaTTD(a,shape,G,t_max): #a ist wieder tau
    beta=a/shape
    
    zeit=np.arange(t_max)

    wertgamma=zeit ** (shape - 1.) / (beta ** (shape) * G) * np.exp(-zeit/beta)
    np.nan_to_num(wertgamma, copy=True)

    plt.plot(zeit,wertgamma)
    plt.show()

    return


### exponential mixing Modell EMM
#unconfined aquifer recieving uniform recharge 
#well samples whole aquifier -> complete mixing (aeschbach folien)
#ines (noch nicht getestet) update: mit Programm getestet -> mit tensordot läuft es
def exponentialMix(a,b,c,inputWerte,t_max,Anzahl_Tracer,Tracer):
    #inputWerte=np.transpose(inputWerte)
    tau1=a
    tau2=b
    ratio=c

    zeit=np.arange(t_max)

    wertex1=1/tau1*np.exp(-zeit/tau1)
    wertex2=1/tau2*np.exp(-zeit/tau2)
    
    wertex1[0] = wertex1[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertex1, copy=True) #test if needed!!
    wertex2[0] = wertex2[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertex2, copy=True)

    intex1=np.sum(wertex1)
    intex2=np.sum(wertex2)

    if intex1 < 0.95 or intex2 < 0.95 or intex1 > 1.05 or intex2 > 1.05:
        print("Achtung, in Exponential Mix wird eine Grenze erreicht! tau1 = ", tau1, "tau2 = ", tau2)
        print("Integral 1 =", intex1, "Integral 2 = ", intex2)
    
    c_calc1=np.zeros(Anzahl_Tracer)  #np.zeros?? dann könnte man eine weitere Variable für die Anzahl an Tracern einfügen
    c_calc2=np.zeros(Anzahl_Tracer)
    c_calc=np.zeros(Anzahl_Tracer)
    k=0
    if 'C14' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'Ar39' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'CFC11' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'CFC12' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    return c_calc
def exponentialMixvary(a,b,c,d,inputWerte,t_max,Anzahl_Tracer,Tracer):
    #inputWerte=np.transpose(inputWerte)
    tau1=a
    tau2=b
    ratio=c

    zeit=np.arange(t_max)

    wertex1=1/tau1*np.exp(-zeit/tau1)
    wertex2=1/tau2*np.exp(-zeit/tau2)
    
    wertex1[0] = wertex1[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertex1, copy=True) #test if needed!!
    wertex2[0] = wertex2[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertex2, copy=True)

    intex1=np.sum(wertex1)
    intex2=np.sum(wertex2)

    if intex1 < 0.95 or intex2 < 0.95 or intex1 > 1.05 or intex2 > 1.05:
        print("Achtung, in Exponential Mix wird eine Grenze erreicht! tau1 = ", tau1, "tau2 = ", tau2)
        print("Integral 1 =", intex1, "Integral 2 = ", intex2)
    
    c_calc1=np.zeros(Anzahl_Tracer)  #np.zeros?? dann könnte man eine weitere Variable für die Anzahl an Tracern einfügen
    c_calc2=np.zeros(Anzahl_Tracer)
    c_calc=np.zeros((Anzahl_Tracer,len(d)))
    k=0
    if 'C14' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'Ar39' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'CFC11' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    if 'CFC12' in Tracer:
         c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
         c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
         c_calc[k]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k]
         k=k+1
    return c_calc
def exponentialMixvary1(a,b,c,inputWerte,t_max,Anzahl_Tracer,Tracer): #variable age 1
    #inputWerte=np.transpose(inputWerte)
    tau1=a
    tau2=b
    ratio=c

    zeit=np.arange(t_max)
    wertex1=np.zeros((len(a),len(zeit)))

    for i in np.arange(len(a)):
        wertex1[i,:]=1/tau1[i]*np.exp(-zeit/tau1[i])

    wertex2=1/tau2*np.exp(-zeit/tau2)
    
    wertex1[:,0] = wertex1[:,1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertex1, copy=True) #test if needed!!
    wertex2[0] = wertex2[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertex2, copy=True)
    c_calc1=np.zeros((Anzahl_Tracer,len(a)))  #np.zeros?? dann könnte man eine weitere Variable für die Anzahl an Tracern einfügen
    c_calc2=np.zeros(Anzahl_Tracer)
    c_calc=np.zeros((Anzahl_Tracer,len(a)))
    k=0
    if 'C14' in Tracer:
        for n in np.arange(len(a)):
            c_calc1[k,n]=np.tensordot(inputWerte[:,k],wertex1[n,:],axes=([0],[0]))
            c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k,n]+(1-ratio)*c_calc2[k]
        k=k+1
    if 'Ar39' in Tracer:
        for n in np.arange(len(a)):
            c_calc1[k,n]=np.tensordot(inputWerte[:,k],wertex1[n,:],axes=([0],[0]))
            c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k,n]+(1-ratio)*c_calc2[k]
        k=k+1
    if 'CFC11' in Tracer:
        for n in np.arange(len(a)):
            c_calc1[k,n]=np.tensordot(inputWerte[:,k],wertex1[n,:],axes=([0],[0]))
            c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k,n]+(1-ratio)*c_calc2[k]
        k=k+1
    if 'CFC12' in Tracer:
        for n in np.arange(len(a)):
            c_calc1[k,n]=np.tensordot(inputWerte[:,k],wertex1[n,:],axes=([0],[0]))
            c_calc2[k]=np.tensordot(inputWerte[:,k],wertex2,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k,n]+(1-ratio)*c_calc2[k]
        k=k+1

    return c_calc
def exponentialMixvary2(a,b,c,inputWerte,t_max,Anzahl_Tracer,Tracer): #variable age 2
    #inputWerte=np.transpose(inputWerte)
    tau1=a
    tau2=b
    ratio=c

    zeit=np.arange(t_max)
    wertex2=np.zeros((len(b),len(zeit)))

    for i in np.arange(len(b)):
        wertex2[i,:]=1/tau2[i]*np.exp(-zeit/tau2[i])

    wertex1=1/tau1*np.exp(-zeit/tau1)
    
    wertex2[:,0] = wertex2[:,1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertex1, copy=True) #test if needed!!
    wertex1[0] = wertex1[1] #so that wertIG1[0] is not zero -> way more likely to be the same as the first time 
    np.nan_to_num(wertex2, copy=True)
    
    c_calc2=np.zeros((Anzahl_Tracer,len(b)))  #np.zeros?? dann könnte man eine weitere Variable für die Anzahl an Tracern einfügen
    c_calc1=np.zeros(Anzahl_Tracer)
    c_calc=np.zeros((Anzahl_Tracer,len(b)))
    k=0
    
    if 'C14' in Tracer:
        for n in np.arange(len(b)):
            c_calc2[k,n]=np.tensordot(inputWerte[:,k],wertex2[n,:],axes=([0],[0]))
            c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k,n]
        k=k+1
    if 'Ar39' in Tracer:
        for n in np.arange(len(b)):
            c_calc2[k,n]=np.tensordot(inputWerte[:,k],wertex2[n,:],axes=([0],[0]))
            c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k,n]
        k=k+1
    if 'CFC11' in Tracer:
        for n in np.arange(len(b)):
            c_calc2[k,n]=np.tensordot(inputWerte[:,k],wertex2[n,:],axes=([0],[0]))
            c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k,n]
        k=k+1
    if 'CFC12' in Tracer:
        for n in np.arange(len(b)):
            c_calc2[k,n]=np.tensordot(inputWerte[:,k],wertex2[n,:],axes=([0],[0]))
            c_calc1[k]=np.tensordot(inputWerte[:,k],wertex1,axes=([0],[0]))
            c_calc[k,n]=ratio*c_calc1[k]+(1-ratio)*c_calc2[k,n]
        k=k+1
    
    return c_calc



###exponential piston flow model EPM
#exponential flow precedes piston flow within the saturated zone 
#segmente of exponential flow followed by a segment of piston flow (tracer LPM)
#ines (noch nicht getestet)
#zwei Parameter -> mean age und EPM ratio (ratio of piston to exponential -> 0 to above 5 from exponential to piston)
def exponentialpiston(a,t_max,inputWerte,EPMratio): #eine Wassermasse a ist tau
    zeit=np.arange(t_max)
    if zeit >= a*(EPMratio/(EPMratio+1)):
        wertepm=(EPMratio+1)/a*np.exp(-(EPMratio+1)*zeit/a + EPMratio)
    else:
        wertepm=0 #loop geht nicht!!!
    #wertepm[0]=wertepm[1] #damit das zeit=0 Problem nicht besteht
    np.nan_to_num(wertepm,copy=True)

    intepm=np.sum(wertepm)

    if intepm < 0.9:
        print("Achtung, in Exponential Piston flow Model wird eine Grenze erreicht! tau=", a)
        print("Integral =", intepm)
    
    c_calc=np.array([0.,0.,0.,0.])

    N=np.arange(4)
    for i in N:
        c_calc[i]=np.dot(inputWerte[:,i],wertepm)
    
    return c_calc 

def epmplot(a,t_max,EPMratio): #eine Wassermasse transit time disribution plot #Problem ist das EMM nur für t >= tau(EPM/(EPM+1)) und sonst 0
    zeit=np.arange(t_max) 
    wertepm=(EPMratio+1)/a*np.exp(-(EPMratio+1)*zeit/a + EPMratio)
    #for zeit 
    wertepm=0
    #wertepm[0]=wertepm[1]   #am Anfang sieht man ein kleines Plateau
    np.nan_to_num(wertepm,copy=True)  #scheint auch zu gehen (ist dann auch nicht 0 am Anfang)

    #plot
    plt.plot(zeit,wertepm)
    plt.show()
    return 


#_______________________________________________________________________________________________________________________________________________________________
'''


hier sind die pymc modelle



'''


import pymc.math as pmm

#exponentialMix mit vier Tracern
def exp(a,b,c,t_max,inputWerte, Tracer):
     tau1=a #a oder astensorvariable(a) scheint keinen Unterschied zu machen
     tau2=b
     ratio=c
     zeit=np.arange(t_max)
     inputWerte=np.transpose(inputWerte)
     #print(np.shape(tau1))
     c_calcAr=0
     c_calcC14=0
     c_calcC11=0
     c_calcC12=0
     c_calc4He=0
     c_calc3H=0

     wertex1=1/tau1*np.exp(-zeit/tau1)
     wertex2=1/tau2*np.exp(-zeit/tau2)

     k=0
     if 'C14' in Tracer:
         c_calc1C14=pmm.dot(inputWerte[k,:],wertex1)
         c_calc2C14=pmm.dot(inputWerte[k,:],wertex2)
         c_calcC14=ratio*c_calc1C14+(1-ratio)*c_calc2C14
         k=k+1
     if 'Ar39' in Tracer:
         c_calc1Ar=pmm.dot(inputWerte[k,:],wertex1)
         c_calc2Ar=pmm.dot(inputWerte[k,:],wertex2)
         c_calcAr=ratio*c_calc1Ar+(1-ratio)*c_calc2Ar
         k=k+1
     if 'CFC11' in Tracer:
         c_calc1C11=pmm.dot(inputWerte[k,:],wertex1)
         c_calc2C11=pmm.dot(inputWerte[k,:],wertex2)
         c_calcC11=ratio*c_calc1C11+(1-ratio)*c_calc2C11
         k=k+1
     if 'CFC12' in Tracer:
         c_calc1C12=pmm.dot(inputWerte[k,:],wertex1)
         c_calc2C12=pmm.dot(inputWerte[k,:],wertex2)
         c_calcC12=ratio*c_calc1C12+(1-ratio)*c_calc2C12
         k=k+1
     if '4He' in Tracer:
         c_calc14He=pmm.dot(inputWerte[k,:],wertex1)
         c_calc24He=pmm.dot(inputWerte[k,:],wertex2)
         c_calc4He=ratio*c_calc14He+(1-ratio)*c_calc24He
         k=k+1
     if '3H' in Tracer:
         c_calc13H=pmm.dot(inputWerte[k,:],wertex1)
         c_calc23H=pmm.dot(inputWerte[k,:],wertex2)
         c_calc3H=ratio*c_calc13H+(1-ratio)*c_calc23H
         k=k+1
     
     return c_calcC14, c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H



#shapefree Mix (rho vor Optimisierung) für vier Tracer

def BinWerte (inputWerte,t_grenzen, Tracer, Anzahl_Tracer): #not used
    # in t_Grenzen werden die Bingrenzen festgelegt. Z.B. ergibt [0, 100, 1000, 20000] 3 Bins

    anzahl_bins = len(t_grenzen) - 1

    rho = np.zeros((Anzahl_Tracer, anzahl_bins))  
    
    n = 0
    k=0                             
    for i in np.arange(anzahl_bins):
        while (n >= t_grenzen[i]) and (n < t_grenzen[i+1]):
            if 'C14' in Tracer:
                c = inputWerte[:,k] / (t_grenzen[i+1] - t_grenzen[i]) 
                rho[k, i] = rho[k, i] + c #rho[k, i] = rho[k, i] + c ValueError: setting an array element with a sequence.
                k=k+1
            if 'Ar39' in Tracer:
                d = inputWerte[:,k] / (t_grenzen[i+1] - t_grenzen[i]) 
                rho[k, i] = rho[k, i] + d
                k=k+1
            if 'CFC11' in Tracer:
                e = inputWerte[:,k] / (t_grenzen[i+1] - t_grenzen[i])
                rho[k, i] = rho[k, i] + e
                k=k+1
            if 'CFC12' in Tracer:
                f = inputWerte[:,k] / (t_grenzen[i+1] - t_grenzen[i])
                rho[k, i] = rho[k, i] + f
                k=k+1
            n += 1
    return rho

def BinWerte2 (inputWerte,t_grenzen, Tracer, Anzahl_Tracer): #sechs tracer ohne zeitschritte
    # in t_Grenzen werden die Bingrenzen festgelegt. Z.B. ergibt [0, 100, 1000, 20000] 3 Bins
    #print(np.shape(inputWerte))

    anzahl_bins = len(t_grenzen) - 1
 
    rhoC14=np.zeros((anzahl_bins))
    rhoAr=np.zeros((anzahl_bins))
    rhoC11=np.zeros((anzahl_bins))
    rhoC12=np.zeros((anzahl_bins))
    rho4He=np.zeros((anzahl_bins))
    rho3H=np.zeros((anzahl_bins))
    rhoNGT=np.zeros((anzahl_bins))
    #print(anzahl_bins)
    print(np.shape(inputWerte))
    #print(rhoC14)
    n = 0                       
    for i in np.arange(anzahl_bins):
        while (n >= t_grenzen[i]) and (n < t_grenzen[i+1]):
            k=0
            if 'C14' in Tracer:
                #c = inputWerte[n,k] / (t_grenzen[i+1] - t_grenzen[i]) 
                #rhoC14[i] = rhoC14[i] + c #rho[k, i] = rho[k, i] + c ValueError: setting an array element with a sequence.
                rhoC14[i] = (t_grenzen[i+1] - t_grenzen[i])/2 + t_grenzen[i] #carbon age
                k=k+1
            if 'Ar39' in Tracer:
                d = inputWerte[n,k] / (t_grenzen[i+1] - t_grenzen[i]) 
                rhoAr[i] = rhoAr[i] + d
                k=k+1
            if 'CFC11' in Tracer:
                e = inputWerte[n,k] / (t_grenzen[i+1] - t_grenzen[i])
                rhoC11[i] = rhoC11[i] + e
                k=k+1
            if 'CFC12' in Tracer:
                f = inputWerte[n,k] / (t_grenzen[i+1] - t_grenzen[i])
                rhoC12[i] = rhoC12[i] + f
                k=k+1
            if '4He' in Tracer:
                g = inputWerte[n,k] / (t_grenzen[i+1] - t_grenzen[i])
                rho4He[i] = rho4He[i] + g
                k=k+1
            if '3H' in Tracer:
                h=inputWerte[n,k] / (t_grenzen[i+1] - t_grenzen[i])
                rho3H[i]=rho3H[i]+h
                k=k+1
            n=n+1
    if 'NGT' in Tracer:
        rhoNGT=np.array([10,9.2,9.3,9.6,2.2,5]) #HO&VO
    return rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT

def shapefree(a,b,c,d,e,f,rhoC14,rhoAr,rhoC11,rhoC12,rho4He,rho3H,rhoNGT, Tracer,t_grenzen): #max 6 bins, min 2
    c_calcC14=0
    c_calcAr=0
    c_calcC11=0
    c_calcC12=0
    c_calc4He=0
    c_calc3H=0
    c_calcNGT=0
    anzahl_bins = len(t_grenzen) - 1
    if anzahl_bins==2:
        aa=pmm.stack((a,b),axis=1)
    if anzahl_bins==3:
        aa=pmm.stack((a,b,c),axis=1)
    if anzahl_bins==4:
        aa=pmm.stack((a,b,c,d),axis=1)
    if anzahl_bins==5:
        aa=pmm.stack((a,b,c,d,e),axis=1)
    if anzahl_bins==6:
        aa=pmm.stack((a,b,c,d,e,f),axis=1)
    k=0
    if 'C14' in Tracer:
        c_calcC14=pmm.dot(rhoC14,aa)
        k=k+1
    if 'Ar39' in Tracer:
        c_calcAr=pmm.dot(rhoAr,aa)
        k=k+1
    if 'CFC11' in Tracer:
        c_calcC11=pmm.dot(rhoC11,aa)
        k=k+1
    if 'CFC12' in Tracer:
        c_calcC12=pmm.dot(rhoC12,aa)
        k=k+1
    if '4He' in Tracer:
        c_calc4He=pmm.dot(rho4He,aa)
        k=k+1
    if '3H' in Tracer:
        c_calc3H=pmm.dot(rho3H,aa)
        k=k+1
    if 'NGT' in Tracer:
        c_calcNGT=pmm.dot(rhoNGT,aa)
        k=k+1
 
    return c_calcC14,c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H,c_calcNGT  


#inverse Gaussian Mix für 5 Tracer mit zeitschritten
def IGMix(a,b,c, inputWerte, t_max1, t_max2,timestep1,timestep2,Tracer): #DMix
    tau1=a
    tau2=b
    ratio=c
    peclet = 10
    c_calcC14=0
    c_calcAr=0
    c_calcC11=0
    c_calcC12=0
    c_calc4He=0
    c_calc3H=0

    #zeitIG = np.arange(t_max)
    zeit1=np.arange(t_max1,step=timestep1)
    zeit2=np.arange(start=t_max1,stop=t_max2,step=timestep2) ####
    zeitIG=np.concatenate((zeit1,zeit2))
    zeitIG[0]=zeitIG[1]
    
    wertIG1 = np.zeros(len(zeitIG))
    wertIG2 = np.zeros(len(zeitIG))
    
    q1 = np.sqrt(peclet * tau1/(4 * np.pi))
    q2 = np.sqrt(peclet * tau2/(4 * np.pi))
  
    wertIG1 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) /
                                                                 (4 * tau1 * zeitIG))
    wertIG2 = q2 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau2) ** 2) /
                                                                 (4 * tau2 * zeitIG))
    k=0
    if 'C14' in Tracer:
        c_calc1C14=pmm.dot(inputWerte[:,k],wertIG1)
        c_calc2C14=pmm.dot(inputWerte[:,k],wertIG2)
        c_calcC14=ratio * c_calc1C14 + (1 - ratio) * c_calc2C14
        k=k+1
    if 'Ar39' in Tracer:
        c_calc1Ar=pmm.dot(inputWerte[:,k],wertIG1)
        c_calc2Ar=pmm.dot(inputWerte[:,k],wertIG2)
        c_calcAr=ratio * c_calc1Ar + (1 - ratio) * c_calc2Ar
        k=k+1
    if 'CFC11' in Tracer:
        c_calc1C11=pmm.dot(inputWerte[:,k],wertIG1)
        c_calc2C11=pmm.dot(inputWerte[:,k],wertIG2)
        c_calcC11=ratio * c_calc1C11 + (1 - ratio) * c_calc2C11
        k=k+1
    if 'CFC12' in Tracer:
        c_calc1C12=pmm.dot(inputWerte[:,k],wertIG1)
        c_calc2C12=pmm.dot(inputWerte[:,k],wertIG2)
        c_calcC12=ratio * c_calc1C12 + (1 - ratio) * c_calc2C12
        k=k+1
    if '4He' in Tracer:
        c_calc14He=pmm.dot(inputWerte[:,k],wertIG1)
        c_calc24He=pmm.dot(inputWerte[:,k],wertIG2)
        c_calc4He=ratio * c_calc14He + (1 - ratio) * c_calc24He
        k=k+1
    if '3H' in Tracer:
        c_calc13H=pmm.dot(inputWerte[:,k],wertIG1)
        c_calc23H=pmm.dot(inputWerte[:,k],wertIG2)
        c_calc3H=ratio * c_calc13H + (1 - ratio) * c_calc23H
        k=k+1

    return c_calcC14, c_calcAr,c_calcC11,c_calcC12,c_calc4He,c_calc3H


#pistonMix
def PMix(a,b,c, inputWerte,t_max,Tracer): #4 Tracer nicht timesteps (Problem-> statt tau index von tau wissen)

    tau1 = a
    tau2 = b
    ratio = c

    if tau1 > t_max or tau2 > t_max:
        print("t_max ist zu klein gewählt! MRT außerhalb der Grenzen!")

    k=0
    if 'C14' in Tracer:
        c_calc1C14=inputWerte[tau1,k]
        c_calc2C14=inputWerte[tau2,k]
        c_calcC14=ratio * c_calc1C14 + (1 - ratio) * c_calc2C14
        k=k+1
    if 'Ar39' in Tracer:
        c_calc1Ar=inputWerte[tau1,k]
        c_calc2Ar=inputWerte[tau2,k]
        c_calcAr=ratio * c_calc1Ar + (1 - ratio) * c_calc2Ar
        k=k+1
    if 'CFC11' in Tracer:
        c_calc1C11=inputWerte[tau1,k]
        c_calc2C11=inputWerte[tau2,k]
        c_calcC11=ratio * c_calc1C11 + (1 - ratio) * c_calc2C11
        k=k+1
    if 'CFC12' in Tracer:
        c_calc1C12=inputWerte[tau1,k]
        c_calc2C12=inputWerte[tau2,k]
        c_calcC12=ratio * c_calc1C12 + (1 - ratio) * c_calc2C12
        k=k+1

    return c_calcC14, c_calcAr,c_calcC11,c_calcC12



#_______________________________________________________________________________________________________________________________________________________________
#transit time distributions
def shapefreeTTD(a,t_grenzen,well_name): 
    t = []
    for i in range(len(t_grenzen) - 1):
        t.append(f"{t_grenzen[i]}-{t_grenzen[i+1]} yr")
    plt.bar(t,a,width=0.5)
    plt.xlabel("Age Bins")
    plt.ylabel("Ratio of each age bin") #how much water is how old
    plt.title("Contributions of the age bins to the water mixture of Well {}".format(well_name[0]))
    plt.show()
    return ()

def shapefreeTTD2(a,err_a,t_grenzen,well_name): #with errors
    t = []
    for i in range(len(t_grenzen) - 1):
        t.append(f"{t_grenzen[i]}-{t_grenzen[i+1]} yr")
    plt.bar(t,a,width=0.5,yerr=err_a,capsize=8)
    plt.xlabel("Age Bins")
    plt.ylabel("Ratio of each age bin") #how much water is how old
    plt.title("Contributions of the age bins to the water mixture of Well {}".format(well_name[0]))
    plt.show()
    return ()

def shapefreeTTD3(a,t_grenzen):
    left=t_grenzen[:-1]
    width = []
    for i in range(len(t_grenzen) - 1):
        width.append(t_grenzen[i + 1] - t_grenzen[i])
    plt.bar(left,a,width,align='edge',label='Shapefree TTD')

def inverseGaussianMixTTD(a,b,c, t_max): #man könnte es hier auch rauslassen
    tau1 = a 
    tau2 =b 
    ratio = c
    peclet = 10 

    zeitIG = np.arange(t_max)
    zeitIG[0] = 0.01 #vielleicht das nicht mit der Zeit 0 gearbeitet wird??
    wertIG1 = np.zeros(len(zeitIG))
    wertIG2 = np.zeros(len(zeitIG))
   
    q1 = np.sqrt(peclet * tau1/(4 * np.pi))
    q2 = np.sqrt(peclet * tau2/(4 * np.pi))
    
    wertIG1 = q1 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau1) ** 2) /
                                                                 (4 * tau1 * zeitIG))
    wertIG2 = q2 * zeitIG ** (-3 / 2) * np.exp((-peclet * (zeitIG - tau2) ** 2) /
                                                                 (4 * tau2 * zeitIG))
    #wertIG1[0]=wertIG1[1]
    #wertIG2[0]=wertIG2[1]
    wertIG=ratio*wertIG1+(1-ratio)*wertIG2

    plt.plot(zeitIG,wertIG)
    plt.xlabel('Time [years]')
    plt.title('Transit Time Distribution - Inverse Gaussian Model')
    plt.show()

    return 

def exponentialMixTTD(a,b,c,t_max):
    zeit=np.arange(t_max)
    wertex1=1/a*np.exp(-zeit/a)
    wertex2=1/b*np.exp(-zeit/b)
    np.nan_to_num(wertex1,copy=True)
    np.nan_to_num(wertex2,copy=True)
    wertex=c*wertex1+(1-c)*wertex2

    plt.plot(zeit,wertex)
    plt.xlabel('Time [years]')
    plt.title('Transit Time Distribution - Exponential Model')
    plt.show()

    return 
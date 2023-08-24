#Input Werte für verschiedene Tracer -> momentan unnötig für Violas Daten einfach inputfunktionen.py von Viola nehmen
import numpy as np
import os
import pandas as pd

directory = os.path.dirname(os.path.abspath(__file__))
ordner=os.path.join(directory, 'grumpy') #weil unterordner
outputPath = os.path.join(directory, 'Output')
inputPath = os.path.join(ordner, 'Input')
inputFile = os.path.join(inputPath, 'Input_functions.xlsx')
measFile = os.path.join(inputPath, 'Datierung_Oman_05_2018.xlsx')
ordner2=os.path.join(directory, 'data') #weil unterordner
tritiumFile = os.path.join(ordner2, 'Tritium_input.xlsx')


Tracer='C14'+'Ar39'+'CFC11'+'CFC12'+'4He'+'3H'+'NGT' #alle Tracer Namen

#für variable Input Werte
def InputWerte(Tracer,Anzahl_Tracer,vogel,t_max,temp,salinity,pressure,excess,deep):
    #Tracer1=True or False ....
    inputWerte=np.zeros(([t_max,Anzahl_Tracer])) #länge wie t_max so ist die Länge von inputTracer auch definiert, variable Anzahl Tracer
    #print(inputWerte)
    k=0
    if 'C14' in Tracer:
        C14Werte=inputC14(vogel,t_max)
        inputWerte[:,k]=C14Werte
        k=k+1
    if 'Ar39' in Tracer:
        Ar39Werte=inputAr39(t_max)
        inputWerte[:,k]=Ar39Werte
        k=k+1
    if 'CFC11' in Tracer:
        zeit,CFC11Werte=inputCFC11(temp,salinity,pressure,excess,t_max)
        inputWerte[:,k]=CFC11Werte
        k=k+1
    if 'CFC12' in Tracer:
        zeit,CFC12Werte=inputCFC12(temp,salinity,pressure,excess,t_max)
        inputWerte[:,k]=CFC12Werte
        k=k+1
    if '4He' in Tracer:
        He4Werte=input4He(deep,t_max)
        inputWerte[:,k]=He4Werte
        k=k+1
    if '3H' in Tracer:
        H3Werte=input3H(t_max)
        inputWerte[:,k]=H3Werte
        k=k+1
    return inputWerte
#man könnte jetzt beliebig mehr Tracer hinzufügen -> eine Ortsfunktion gibt es noch nicht (weil oft Tracer je nach Ort unterschiedliche Input Funktionen haben)
    
    



#Input Funktion für Argon 39 (Anywhere)
def inputAr39(t_max):
    decay = 0.002576755  # Ar39 Zerfallskonstante gamma in 1/Jahr
    inpZeit   = np.arange(t_max)
    inpWert   = np.zeros(len(inpZeit))
    
    
    for n in range(len(inpZeit)):
            inpWert[n] = 100 * np.exp(-decay * inpZeit[n])
            
    return inpWert
#inpWert=inputAr39(t_max)

#Input Funktion C14
def inputC14(vogel, t_max): #für Viola sampling 2018, für niederlande 2020
    decay = 0.000120968  # C14 Zerfallskonstante gamma in 1/Jahr

    #c14 = pd.read_excel(inputFile, sheet_name='C14', usecols='A, B').values
    #wert = c14[:, 1]  # wert: atm. C14-Konzentration in pmC
    wert=([102.000,102.300,102.600,102.900,103.200,103.500,103.800,104.100,104.400,104.700,105.000,105.300,105.600,105.900,106.200,106.500,106.800,107.100,107.400,107.700,108.400,109.000,109.700,110.300,110.900,111.700,111.900,112.600,113.700,114.500,115.400,116.400,117.500,118.700,119.400,120.600,121.300,122.900,124.200,126.100,127.200,129.700,132.500,133.500,135.200,138.300,140.800,143.600,147.200,149.600,151.700,154.300,156.200,160.300,163.500,168.700,169.900,154.900,125.800,120.200,119.500,115.100,110.300,103.600,100.500,98.900])
    #wert kopiert aus Exel -> um Exel aufrufen zu umgehen
    inpZeit = np.arange(t_max)
    inpWert = [100] * len(inpZeit)
    # Fuer die letzten 63 Jahre vor 2018 wird nicht 100pmC angenommen, sondern die Werte aus Excel (Bomb Peak...)
    if t_max > 64:
        for n in range(0, 64):
            inpWert[n] = wert[n]  # Samplingjahr: 2018 (+2), erster Input-Eintrag: 2020
    else:
        for n in range(0, int(t_max)):
            inpWert[n] = wert[n]  # Samplingjahr: 2018, erster Input-Eintrag: 2020

    if vogel:
        for n in range(len(inpZeit)):
            inpWert[n] = inpWert[n] * 0.8 * np.exp(-decay*inpZeit[n])
    else:
        for n in range(len(inpZeit)):
            inpWert[n] = inpWert[n] * np.exp(-decay*inpZeit[n])
        
    return inpWert

def inputCFC11(temp, salinity, pressure, excess, t_max):
    #Excess Air in cc/kg
    #Druck in bar

    temp = int(temp + 273.15) #Umrechnung von Grad C in Kelvin
    
    #Koeffizienten zur Berechnung der Loeslichkeit von CFC11 (Warner und Weiss, 1985)
    a1 = -136.2685
    a2 = 206.115
    a3 = 57.2805
    b1 = -0.148598
    b2 = 0.095114
    b3 = -0.0163396
    
    #Berechne Loeslichkeit
    henry = np.exp(a1+a2*(100/temp)+a3*np.log(temp/100)+salinity*(b1+b2*(temp/100)+b3*(temp/100)**2))
    #print("Henry Coeff von CFC11: ", henry)
    cfc11 = pd.read_excel(inputFile, sheet_name='CFC11', usecols=('A,B')).values
    #zeit: Jahreszahl 
    zeit    = cfc11[:, 0]
    #wert: atm. CFC11-Konzentration in ppt
    wert    = cfc11[:, 1]
    inpZeit = np.arange(t_max)
    inpWert = np.zeros(len(inpZeit))

    #inpWert: geloeste Konzentration im Wasser (in 10^(-9) mol/kg)
    if t_max >= 79:
        for n in range(0, 79):
            inpWert[n] = wert[n]*pressure*(henry+excess/22414.1)
    else:
        for n in range(0, int(t_max)):
            inpWert[n] = wert[n]*pressure*(henry+excess/22414.1)

    return inpZeit, inpWert


def inputCFC12(temp, salinity, pressure, excess, t_max): #2do: anpassen auf 2018!
    # Excess Air in cc/kg
    # Druck in bar
    # Umrechnung von Celsius in Kelvin:
    temp = int(temp + 273.15)
    
    # Koeffizienten zur Berechnung der Loeslichkeit von CFC12
    a1 = -124.4395
    a2 = 185.4299
    a3 = 51.6383
    b1 = -0.149779
    b2 = 0.094668
    b3 = -0.0160043
    
    # Berechne Loeslichkeit
    henry = np.exp(a1+a2*(100/temp)+a3*np.log(temp/100)+salinity*(b1+b2*(temp/100)+b3*(temp/100)**2))
    #print("Henry Coeff von CFC12: ", henry)
    cfc12 = pd.read_excel(inputFile, sheet_name='CFC12', usecols='A, B').values
    
    # zeit: Jahreszahl
    zeit    = cfc12[:, 0]
    # wert: atm. CFC12-Konzentration in ppt
    wert    = cfc12[:, 1]
    inpZeit = np.arange(t_max)
    inpWert = np.zeros(len(inpZeit))
    
    # inpWert: geloeste Konzentration im Wasser (in 10^(-9) mol/kg)
    if t_max >= 76:
        for n in range(3, 76):
            inpWert[n] = wert[n-3]*pressure*(henry+excess/22414.1)
    else:
        for n in range(3, int(t_max)):
            inpWert[n] = wert[n-3]*pressure*(henry+excess/22414.1)

    return inpZeit, inpWert

def input4He(deep,t_max):
    Zeit=np.arange(t_max)
    if deep==True:
        #accumulation rate 1.5e-11 ccSTG/g per year
        Wert=Zeit*1.5e-11
    else:
        #accumulation rate 4.76e-11 ccSTG/g per year
        Wert=Zeit*4.76e-11 
    return Wert

#excel file from tracer lpm for 47-49, <96 (coordinates) due to lack of other input values
def input3H(t_max): #samplingyear 2020 (VO,VT,HO), for BW it should be 2014
    wert=pd.read_excel(tritiumFile, usecols='B',skiprows=20).values.flatten()[::2]
    zeit=pd.read_excel(tritiumFile, usecols='A',skiprows=20).values.flatten()[::2]
    if t_max >= 130:
        wert = np.concatenate([wert[:130], np.full(t_max-130, 13.1659686358031)])
    else:
        wert=wert[:t_max]
    decay = 0.056353429 #Tritium Zerfallskonstante decay=ln(2)/12.3 years
    Zeit = np.arange(t_max)
    inpWert=np.zeros(t_max)
    for n in range(len(Zeit)):
            inpWert[n] = wert[n] * np.exp(-decay * Zeit[n])

    return inpWert

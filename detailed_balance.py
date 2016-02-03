# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 16:10:41 2015

@author: spolly
"""

import numpy as np
from scipy import integrate, special, optimize, constants as sp
from numba import jit


def N(Emin,Emax,T,mu): #emission flux from photovoltaic
    N0,N0err=integrate.quad(E,Emin,Emax,args=(T,mu))
    return 2*np.pi/((sp.h/sp.e)**3*sp.c**2)*N0

@jit 
def E(myE,myT,myMu): #energy function to integrate for flux (N)
    return myE**2/(np.exp((myE-myMu)/((sp.k/sp.e)*myT))-1)

def L(T):
    L0,L0err=integrate.quad(lambda E: E**3/(np.exp((E)/((sp.k/sp.e)*T))-1),lE(T)/sp.e,uE(T)/sp.e)
    return 2*np.pi/((sp.h/sp.e)**3*sp.c**2)*L0

def fw(R,L):
    ws=(np.pi*(R*2/L)**2)/4 #steradians
    return ws/np.pi #tecnically (ws*cos(theta))/pi where theta is incident angle


def uE(T): #[J] upper energy limit to not break quad integrating to +inf
    
#    #This can be solved analytically, but like most things relating to Planck's
#    #equation, there exists a simple linear relationship with temperature
#    #which was solved using the below method:
#    hv=planckPeak(T) #[J]
#    while np.abs((hv/(k*T)-500)) > 1: #Contents of np.exp() nearly 500
#        if hv/(k*T) > 500:
#            hv=hv-hv/2 #[J]
#        else:
#            hv=hv+hv/2 #[J]
#    return hv #[J]
#    #this linear equation reduces computation time by ~3 orders of magnitude:
    
    return 6.89699e-21*T+6.16297e-33 #[J]

def planckPeak(T): #[J]
    return np.real((5+special.lambertw(-5/np.e**5))*sp.k*T) # [J]

def lE(T): #Upper limit to not break quad integrating to +inf
    return planckPeak(T)/1e4 #[J] 1e4 higher than the peak gives a low error

def pMax(T):
    return (2*(np.pi**5)*(sp.k*T)**4)/(15*sp.h**3*sp.c**2)
      
def maxEffGlobal(Ts,Tc,Ps,fs,X,n):
    #Eg0=np.linspace(0.5*n,0.5,n)
    #myEg=optimize.differential_evolution(lambda x:-maxEff(x,Ts,Tc,Ps,fs,X,uLim)[0], Eg0, method='Nelder-Mead',options={'disp': True, 'maxiter':100000})
    myRange=[(0,12)]*n    
    #myEg=optimize.differential_evolution(lambda x:-maxEff(x,Ts,Tc,Ps,pfs,X,uLim)[0], myRange, maxiter=5000, popsize=15, tol=0.0001)
    myEg=optimize.differential_evolution(maxEff, myRange, args=(Ts,Tc,Ps,fs,X), maxiter=100000, popsize=15, tol=0.0001)
    if myEg['success']:
        return (myEg)#,maxEff(myEg.x,,myEg.y,Ts,Tc,Ps,fs,X,uLim),maxEff(myEg.y,Ts,Tc,Ps,fs,X,uLim))   
    else:
        return (myEg) 
        
def maxEff(Eg,Ts,Tc,Ps,fs,X):
    uLim=uE(Ts)/sp.e
    if (Eg.size > 1) & all(Eg[i] <= Eg[i+1] for i in range(Eg.size-1)):
        #check to see that all Eg values are in order from high to low
        #if not, it is a bad set of bandgaps and the function returns null
        return (0)
    else:
        myJ=np.zeros(Eg.size)
        myV=[]
        preFactor=np.zeros(Eg.size)
        
        for i in range(Eg.size):
            if i==0:
                Emax=uLim
            else:
                Emax=Eg[i-1]
            preFactor[i]=X*fs*N(Eg[i],Emax,Ts,0)+(1-X*fs)*N(Eg[i],Emax,Tc,0)
            #myV.append(optimize.minimize_scalar(lambda V:-((sp.e*(preFactor[i]-N(Eg[i],uE(Tc)/sp.e,Tc,V)))*V/Ps),bounds=(0,Eg[i]),method='bounded',options={'disp': True}))
            myV.append(optimize.minimize_scalar(V, bounds=(0,Eg[i]), args=(preFactor[i],Eg[i],Tc,Ps), method='bounded',options={'disp': True}))
 
        if all(myV[i].success for i in range(Eg.size)): 
            for i in range(Eg.size):
                myJ[i]=(sp.e*(preFactor[i]-N(Eg[i],uE(Tc)/sp.e,Tc,myV[i].x)))
            minJ=myJ.min()
            totalV=sum(myV[i].x for i in range (Eg.size))
            myEff=minJ*totalV/Ps
            return (-myEff)   
        else:
            return (float('NaN'))
            
def V(myV,myPreFactor,myEg,myTc,myPs):
    return -((sp.e*(myPreFactor-N(myEg,uE(myTc)/sp.e,myTc,myV)))*myV/myPs)



#Earth
Ts=6000                     #[K] sun temperature 5775
Tc=300                      #[K] pv temperature

solar_radius=6.96342e8*1    #[m] sun radius
sR=1*solar_radius           #multiplier for star radius

au=1.496e11*1               #[m] orbital distance of pv from sun
sma=1*au                    #semi-major axis in terms of au

vF=fw(sR,sma)
C=1                     #concentration factor (max is 1/vf)
Pmax=C*vF*L(Ts)*sp.e    #max power from star at planet (W/m^2)
n=3                     #number of junctions to in a multijunction to calculate

junctions=maxEffGlobal(Ts,Tc,Pmax,vF,C,n)
print(junctions)

#!/usr/bin/env python3

# Copyright (c) Météo France (2025-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info

# Le but de ce fichier est de mettre toutes les fonctions qu'on peut potentiellement appelés pour faire tourner nos futur codes
# Ensuite il suffira de faire un appelle du fichier et de la fonction qu nous interesse
# Exemple avec cette fonction test que j'appelle dans le code Sedimentation.py

### Imports ###

from math import *
import time
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.integrate import quad
from scipy.optimize import brentq
from collections import deque
from matplotlib.ticker import MultipleLocator
from matplotlib.ticker import MaxNLocator

### Classes and functions definition ###

class Eq :
    def __init__(self,esp):
        if esp=="i":
            self.a=0.82
            self.b=2.5
            self.c=800
            self.d=1
            self.alpha=3
            self.nu=3
        if esp=='s':
            self.a=0.02
            self.b=1.9
            self.c=5.1
            self.d=0.27
            self.alpha=1
            self.nu=1
        if esp=='g':
            self.a=19.6
            self.b=2.8
            self.c=124
            self.d=0.66
            self.alpha=1
            self.nu=1
        if esp=='r':
            self.a=524
            self.b=3
            self.c=842
            self.d=0.66
            self.alpha=1
            self.nu=1
        if esp=='c':
            self.a=524
            self.b=3
            self.c=3.2e7
            self.d=2
            self.alpha=1
            self.nu=3         

    def Gamma(self, diametre, lam) :
        return (self.alpha/gamma(self.nu))*(lam**(self.alpha*self.nu))*(diametre**(self.alpha*self.nu-1))*np.exp(-((lam*diametre)**self.alpha))
   
    def G(self, p) :
        return (gamma(self.nu+p/self.alpha)/gamma(self.nu))
    
    def Lanbda(self, rho_r, N):
        return (((rho_r)/(self.a*N*self.G(self.b)))**(-1/self.b))
    
    def Masse(self, diametre):
        return (self.a*(diametre**self.b))
    
    def Vitesse(self,diametre):
        return(self.c*(diametre**self.d))
    
    def Calcul_rho_r(self, m, n):
        return np.dot(m,n)
    
    def Dmin_Dmax(self, lam):
 
        def F(D):
            return quad(self.Gamma, 0, D, args=(lam))[0]
 
        D_high = 1.0 / lam # échelle naturelle
        while F(D_high) < 0.999:
            D_high *= 2

        Dmin = brentq(lambda D: F(D) - 0.01, 0, D_high)
        Dmax = brentq(lambda D: F(D) - 0.99, 0, D_high)
 
        return Dmin, Dmax
    
    def Classe_D(self, nb_classes, Dmin, Dmax, N, lam):
        Result=[]
        Intervalle=(Dmax-Dmin)/nb_classes
        for i in range(nb_classes):
            Di=(1+2*i)*Intervalle/2 + Dmin
            
            P_i=quad(self.Gamma, Dmin+i*Intervalle, Dmin+(i+1)*Intervalle, args=(lam))[0]/0.98
            #print(i, P_i)
            Ni=N*P_i
            Result.append([Di, Ni]) #Liste de deux paramètres : diamètre moyen, quantité associé par rapport au nombre total de particule.
        return Result

class Affichage :

    def Affichage_Concentration(Concentration, typ, model): #type="concentration" ou "masse"
        Temps_simu=len(Concentration)
        nb_boites=len(Concentration[0])
        #time=np.linspace(1, Temps_simu, Temps_simu)
        Concentration=np.array(Concentration)
        Transpose=Concentration.T
        #print(Temps_simu, nb_boites)
        plt.figure(figsize=(Temps_simu, nb_boites))
        orig_map=plt.cm.get_cmap('gist_ncar')
        reversed_map = orig_map.reversed()
        plt.pcolormesh(Transpose,cmap=reversed_map)
        plt.title(f"Evolution de la {typ} de particules dans le temps")
        plt.xlabel("Temps")
        plt.ylabel("Mailles du modèle")
        plt.colorbar()
        plt.savefig(f"./fig/{model}/{typ}.png")
        plt.show()


    def Affichage_Precipitation(Precip, model):
        Precip=np.array(Precip)
        liste=np.zeros(len(Precip))
        Cumul=[]

        for i in range(len(Precip)):
            liste[i]=1
            Cumul.append(np.dot(Precip, liste))
        print(np.dot(Precip, liste))
        fig, ax1 = plt.subplots()
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MaxNLocator(10))

        time=np.linspace(1, len(Precip), len(Precip))
        time2 = time -(time[1]-time[0])/2
        ax1.bar(time2, Precip, color="blue", label="Précip_par_pas_de_temps")
        ax1.set_xlabel("temps")
        ax1.set_ylabel("Cumul")
        ax2 = ax1.twinx()
        ax2.plot(time, Cumul, '--', color="red", label="Cumul")
        ax2.set_ylabel('Précip_par_pas_de_temps')
        plt.title("Evolution des précipitations par pas de temps et cumulée")
        plt.grid(axis='x', which='major', markevery=[1,2,3],lw=2, ls=':')
        fig.legend(loc=2)
        plt.savefig(f"./fig/{model}/Précipitations.png")
        plt.show()





def gaussienne (m, sigma,h):
        return (1/(sigma*np.sqrt(2*np.pi))) * exp (-1/2 * ((h-m)/sigma)**2)

class InitialCond :
    '''
    Class that poduces a dataset to initialize the sedimentation model, a type of initialization is used in a restricted area of the vertical gridding.
    The dataset contains the concentration profile of the specie for each bin of the sedimentation scheme.

    Parameters
    ----------
    nb_grid : number of grid cells in the model
    esp : specie considered "i", "s", "g", "r" or "c"
    mode : type of intial condition used, by default the last grid cell contains a relative concentration of 1
    Hmax : atltitude of the maximum of the gaussian fonction of the concentration if the type of initial condition is "gauss"
    sigma : value of the sigma parameter used for the gaussian function if the type of initial condition is "gauss"
    

    Attributes
    ----------
    self.levels_boundaries : list containing the altitudes of the boundaries of each grid mesh point
    self.grid : list containing the grid points
    self.data : DataArray having for variables
        concentration_bin_{i} : the concentrations of the specie in the i-th bin over the vertical grid
        diameter_bin_{i} : the diameter of the specie for the i-th bin
    and for coordinates 
        level : the levels of the grid points

    '''
    
    def __init__(self, nb_grid, esp, mode = "simple", Hmax = 5000, sigma = 2000, nb_classes = 10, rho_r = 1, N = 1) :

        if nb_grid == "ARO" : 
            self.levels_boundaries = [5.00148256575414, 16.7609146275979, 31.9999856716034, 50.6506387418972, 72.6448134875948, 97.9144556307367, 126.391508411840, 158.007915671418, 192.695621996127, 230.386571302649, 271.012706897240, 314.505973414367, 360.798314810016, 409.821674828922, 461.507997847950, 515.890042408161, 573.093937517738, 633.231198491812, 696.398736766220, 762.678848044096, 832.139215500780, 904.832909766711, 980.798389785428, 1060.05950095623, 1142.62547636213, 1228.49093654830, 1317.63588971212, 1410.02573054417, 1505.73432230725, 1604.93984760156, 1707.79812299161, 1814.44258334521, 1924.98428740911, 2039.51193175523, 2280.76794378598, 2407.56182949566, 2538.47269089309, 2673.47735336959, 2812.53026865253, 2955.56351459630, 3102.48360541930, 3253.19498943504, 3407.62661862652, 3565.73195622143, 3727.48898443499, 3892.90019410005, 4061.99258757885, 4234.81767907587, 4411.45149612596, 4591.99457746971, 4776.57197432233, 4965.33325159684, 5158.45249292134, 5356.12828712931, 5558.65852457596, 5766.45816179863, 5980.00284181189, 6199.82890110527, 6426.53336977965, 6660.77397670184, 6903.26915353359, 7154.79804227343, 7416.20049682756, 7688.37709125546, 7972.28912861375, 8268.07660797884, 8575.22917035383, 8893.62412699775, 9223.52646472597, 9565.58886389478, 9920.85173976858, 10290.7432801852, 10677.0795176319, 11082.0546910758, 11510.3223903101, 11966.1795634141, 12454.7528314299, 12982.0128918015, 13554.6557230282, 14180.1025765563, 14866.4999945363, 15627.4731252487, 16485.0471290191, 17467.8152385289, 18610.9387773901, 19955.6070022098, 21550.0160473017, 23450.1477297912, 34461.1214067193]
            nb_grid = 89
        else :
            height_grid = 12e3
            self.levels_boundaries = np.linspace(1/nb_grid * height_grid, height_grid, nb_grid)
        nb_levels = len (self.levels_boundaries)

        boundaries = deque(self.levels_boundaries)
        boundaries.appendleft(0)
        
        self.grid = [(boundaries[i]+boundaries[i+1])/2 for i in range(len(boundaries)-1)]

        if mode == "simple" :
            concentration_profile = [0 for i in range (nb_levels)]
            concentration_profile [-1] = 1
        if mode == "gauss" :
            concentration_profile = [gaussienne(Hmax, sigma, self.grid[i]) for i in range(nb_grid)]

        self.rho_r_profile = np.array(concentration_profile) * rho_r

        eq=Eq(esp)
        lam = eq.Lanbda (rho_r, N)
        dmin, dmax = eq.Dmin_Dmax(lam)
        self.bin_concentration = eq.Classe_D (nb_classes, dmin, dmax, N, lam) # division in n bins

        bin_profile = [np.array(concentration_profile) * self.bin_concentration[i][1] for i in range(len(self.bin_concentration))] # computinng of the n bin profiles
        data_vars1 = {f"concentration_bin_{ind_bin+1}" : ("level", bin_profile[ind_bin]) for ind_bin in range(len(self.bin_concentration))}

        data_vars2 = {f"diameter_bin_{ind_bin+1}" : self.bin_concentration[ind_bin][0] for ind_bin in range(len(self.bin_concentration))} # addition of the diameters
        data_vars1.update(data_vars2)

        self.data = xr.Dataset(data_vars= data_vars1, coords = {"level" : self.grid})

### Tests and verifications ###

#start = time.time()
#initial_conds = InitialCond(nb_grid = "ARO", esp = "r", mode = "gauss", nb_classes = 10, rho_r = 100, N = 10)

#print (initial_conds.levels_boundaries)
#print (initial_conds.grid)
#print (initial_conds.data)
#print (initial_conds.data["concentration_bin_1"])
#print (initial_conds.data["level"])

#z = initial_conds.grid
#for i in range(10): 
    #c = initial_conds.data[f"concentration_bin_{i+1}"]
    #plt.plot(c,z)
#plt.show()

#d = [float(initial_conds.data[f"diameter_bin_{i+1}"]) for i in range(10)]
#x = np.linspace(1,10,10)
#plt.plot(x,d)
#plt.show()

#!/usr/bin/env python3

# Copyright (c) Météo France (2025-)
# This software is governed by the CeCILL-C license under French law.
# http://www.cecill.info

# We import the necessary libraries as well as the various models.
import time
import argparse
import numpy as np
from ditribution import distribution



# Here we define the functions so that the parsers are properly entered.

def check_model(model):
    """
    function that takes the model provided by the user as input, checks its compliance, and returns an appropriate response
    nothing if compliant, 
    error message otherwise
    """
    poss_model = ['Box_Lagrangien','Semi_Lagrangian','EULE','EULE2','STAT']
    if model not in poss_model:
        raise argparse.ArgumentTypeError(f"{model} is not an accepted model, {poss_model} are.")
    return model

def check_advance(advance):
    """
    function that takes as input the type of advancement given by the user, checks its compliance, and returns an appropriate response
    nothing if compliant, 
    error message otherwise
    """
    poss_advance = ['Step_Forward','Step_By_Step']
    if advance not in poss_advance:
        raise argparse.ArgumentTypeError(f"{advance} is not an accepted advancement method, {poss_advance} are.")
    return advance

def check_numb(number):
    """
    function that takes as input the number of stitches or bins given by the user, checks its compliance, and returns an appropriate response
    nothing if compliant, 
    error message otherwise
    """
    try:
        number = int(number)
    except:
        raise argparse.ArgumentTypeError(f"{number} is not an integer.")
    if number<1:
        raise argparse.ArgumentTypeError(f"{number} is not positif.")
    return number

def check_deformable(answer):
    """
    function that takes as input the answer for deformability given by the user, checks its compliance, and returns an appropriate response
    nothing if compliant, 
    error message otherwise
    """
    if answer not in ['Yes','No']:
        raise argparse.ArgumentTypeError(f"You have to answer by Yes or No for deformability.")
    return answer

def check_efficiency_test(answer):
    """
    function that takes as input the answer for efficiency_test given by the user, checks its compliance, and returns an appropriate response
    nothing if compliant, 
    error message otherwise
    """
    if answer not in ['Yes','No']:
        raise argparse.ArgumentTypeError(f"You have to answer if you want to use an efficiency test by Yes or No.")
    return answer

def check_specie(specie):
    """
    function that takes as input the type of advancement given by the user, checks its compliance, and returns an appropriate response
    nothing if compliant, 
    error message otherwise
    """
    accepted_species = ['i', 's', 'g', 'r','c']
    if specie not in accepted_species:
        raise argparse.ArgumentTypeError(f"{specie} is not an accepted specie, {accepted_species} are.")
    return specie



# We provide text to help the user enter their settings.

parser = argparse.ArgumentParser(
    prog = "prediction",
    description = "This code launches the simulation of a sedimentation model according to the given settings." \
    "It requires the type of model, the method of advancement, the number of stitches,"\
    "and other options specific to the chosen model.""",
    epilog = "It then returns a sedimentation display on matplotlib."
)



# We collect the different options selected by the user.

parser.add_argument('-m','--model', help = "Choose between Box_Lagrangien, Semi_Lagrangien, EULE, EULE2 and STAT schemes.", default = "Box_Lagrangien", type=check_model)
parser.add_argument('-s','--type_advance',help="Choose between Step_Forward and Step_By_Step", default="Step_By_Step",type=check_advance)
parser.add_argument('-n','--number_stitches', help = "Pick a positif integer for the number of stitches", default = 100 , type=check_numb)
parser.add_argument('-d','--deformable', help = "Choose bewteen Yes and No (deformable or not)", default="No", type=check_deformable)
parser.add_argument('-b','--number_bin', help = "Pick a positif integer for the number of bin", default=2, type=check_numb)
parser.add_argument('-N','--number_particules', help = "Pick a positif integer for the number of particules", default=100, type=check_numb)
parser.add_argument('-t','--time_step', help = "Pick a positif integer for the time step", default=10, type=check_numb)
parser.add_argument('-c','--CFL', help = "Choose bewteen Yes and No (CFL repected or not)", default='No', type=check_deformable)
parser.add_argument('-S','--speed_max', help = "Pick a positif integer for maximum speed", default=1000, type=check_numb)
parser.add_argument('-e','--specie', help = "Pick an hydrometeor specie, accepted species are 'i', 's', 'g', 'r' or 'c'", default="r", type=check_specie)
parser.add_argument('-f','--efficiency_test', help = "Used as a developpement tool to compare differents versions of the models (choose between Yes and No)", default="No", type=check_efficiency_test)

model = parser.parse_args().model
type_advance = parser.parse_args().type_advance
number_stitches = parser.parse_args().number_stitches
deformable = parser.parse_args().deformable
number_bin = parser.parse_args().number_bin
number_particules = parser.parse_args().number_particules
time_step = parser.parse_args().time_step
CFL = parser.parse_args().CFL
speed_max = parser.parse_args().speed_max
esp = parser.parse_args().specie
efficiency_test = parser.parse_args().efficiency_test


# We print its choices
print(model,type_advance,number_stitches,deformable,number_bin,number_particules,time_step,speed_max,esp,CFL,efficiency_test)

distribution(model,type_advance,number_stitches,deformable,number_bin,number_particules,time_step,speed_max,esp,CFL,efficiency_test)

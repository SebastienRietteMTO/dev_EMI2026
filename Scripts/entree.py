
# We import the necessary libraries as well as the various models.
import time
import argparse
from box_lagrangien import Model



# Here we define the functions so that the parsers are properly entered.

def check_model(model):
    """
    function that takes the model provided by the user as input, checks its compliance, and returns an appropriate response
    nothing if compliant, 
    error message otherwise
    """
    poss_model = ['Box_Lagrangien','Semi_Lagrangian']
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
    function that takes as input the number of stitches given by the user, checks its compliance, and returns an appropriate response
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




# We provide text to help the user enter their settings.

parser = argparse.ArgumentParser(
    prog = "prediction",
    description = "This code launches the simulation of a sedimentation model according to the given settings." \
    "It requires the type of model, the method of advancement, the number of stitches,"\
    "and other options specific to the chosen model.""",
    epilog = "It then returns a sedimentation display on matplotlib."
)



# We collect the different options selected by the user.

parser.add_argument('-m','--model', help = "Choose between Box_Lagrangien and Semi_Lagrangien Scheme.", default = "Box_Lagrangien", type=check_model)
parser.add_argument('-s','--type_advance',help="Choose between Step_Forward and Step_By_Step", default="Step_By_Step",type=check_advance)
parser.add_argument('-n','--number_stitches', help = "Pick a positif integer for the number of stitches", default = 3 , type=check_numb)
parser.add_argument('-d','--deformable', help = "Choose bewteen Yes and No (deformable or not)", default='No', type=check_deformable)

model = parser.parse_args().model
type_advance = parser.parse_args().type_advance
number_stitches = parser.parse_args().number_stitches
deformable = parser.parse_args().deformable



# We print its choices

print(model,type_advance,number_stitches,deformable)

if model == 'Box_Lagrangien':

    # On appelle le code qui gère le modèle Box-Lagrangien en l'initialisant vec les paramètres rentré par l'utilisateur

    model_config = Model(type_advance,number_stitches,deformable)

    profil = model_config.run()

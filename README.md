

This code is used in the following publication [(preprint tba)](url).

[1] Alea Schröder, Steffen Gracla, Maik Röper, Dirk Wübben, Carsten Bockelmann, Armin Dekorsy
"Flexible Robust Beamforming for Multibeam Satellite Downlink using Reinforcement Learning"
to be submitted to *IEEE ICC 2024*.

Email: {**schroeder**, **gracla**, roeper, wuebben, bockelmann, dekorsy}@ant.uni-bremen.de

The project structure is as follows
```
root
|   .gitignore  # gitignore
|   README.md  # this file
|   requirements.txt  # project dependencies
|           
+---models  # trained models & their configurations
|   +---1_sat_10_ant_3_usr_100000_dist_0.05_error_on_cos_0.1_fading  # rSAC1
|   |                       
|   +---1_sat_10_ant_3_usr_100000_dist_0.0_error_on_cos_0.1_fading  # pSAC1
|   |                       
|   +---1_sat_16_ant_3_usr_10000_dist_0.0_error_on_cos_0.1_fading  # pSAC3
|   |                       
|   +---1_sat_16_ant_3_usr_10000_dist_0.05_error_on_cos_0.1_fading  # rSAC3
|   |                       
|   +---1_sat_16_ant_3_usr_100000_dist_0.05_error_on_cos_0.1_fading  # rSAC2
|   |                       
|   \---1_sat_16_ant_3_usr_100000_dist_0.0_error_on_cos_0.1_fading  # pSAC2
|               
+---reports  # figures & supplementary plots
|
+---src  # project code
|   +---analysis  # related to evaluation
|   |
|   +---config  # configuration files
|   |           
|   +---data  # related to satellite communication simulation
|   |           
|   +---models  # related to learning models
|   |           
|   +---plotting  # plotting functions
|   |           
|   +---tests  # code tests
|   |           
|   \---utils  # shared helper functions
```

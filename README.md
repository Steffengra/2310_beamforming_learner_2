
This code was used in the paper "Learning Model-Free Robust Precoding for Cooperative Multibeam Satellite Communications"
by Steffen Gracla, Alea Schröder, Maik Röper, Carsten Bockelmann, Dirk Wübben, Armin Dekorsy
submitted to the
SDPNGS2023: Signal and Data Processing for Next Generation Satellites Workshop of the
IEEE 2023 IEEE International Conference on Acoustics, Speech and Signal Processing.

Email: {**gracla**, schroeder, roeper, bockelmann, wuebben, dekorsy}@ant.uni-bremen.de


#### Folder Structure

|                  |                                                  |
|------------------|--------------------------------------------------|
| models           | trained models or summaries                      |
| outputs          | outputs produced by src                          |
| reports          | generated analysis, latex etc.                   |
| └─ figures       |                                                  |
| src              |                                                  |
| ├─ analysis      | results evaluation                               |
| ├─ config        | configuration files                              |
| ├─ data          | data generation, loading, etc.                   |
| ├─ models        | model training and inference                     |
| ├─ plotting      | plotting functions                               |
| └─ utils         | functions shared between different parts of code |
| .gitignore       | .gitignore                                       |
| README.md        | readme                                           |
| requirements.txt | package requirements                             |

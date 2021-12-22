# CS-433 - Machine learning projet 2

## Introduction
The aim of this project is to learn to use the concepts of machine learning presented in the lectures and practiced in the labs on a
real-world dataset. For this project we chose to collaborate with an EPFL lab TRANSP-OR who provided a historical traffic dataset of a bridge in Switzerland, in order to generate discrete traffic data using ML methods.

## Organisation
This project is organized as follows :

- the repository **_src_** that includes: 
    - **_0-sumo.ipynb_** that briefly shows how to simulate traffic using "Simulation of Urban MObility" (SUMO) of our dataset.
    - **_1-data-exploration.ipynb_** were the data exploration of our datasat is made.
    - **_2-forecasting-model-selection.ipynb_** that selects a model to predict the hourly number of cars per week.
    - **_3-sampling-interval-selection.ipynb_** that finds the most appropriate sampling interval for which to predict the number of cars.
    - **_4-rate-prediction.ipynb_** that predicts the number of vehicules, speed and weight per hour that we called "rate".
    - **_5-discrete-event-generation.ipynb_** that converts the previous rate into discrete events.
    - **_helper-functions.py_** that contains the pipeline and helper functions.
    - several .xml files that helps to define the road to generate the traffic on SUMO.
    - **_test.sumocfg_** that generates a traffic simulation of the dataset on SUMO.   
- the file **_ML-Project-2.pdf_** which is our report that provides a full explanation of our ML system and our findings.

Dire les librairies utilisés, run nos notebooks de manière successive de 1 à 5, mettre un lien de notre dataset
## How to use our project
Just make sure to have the libraries mentioned below installed on your environment before running the cells in the jupyter notebook.
To reproduce our setup, please run the notebooks in a successive way (from 1 to 5).
Don't forget to put the dataset on a repository "data" at the same level of the repository "src". You can find our dataset named "405.txt" on this [link](https://drive.switch.ch/index.php/s/190lRT2jVT5bCgJ).

## Libraries
In this project we used these libraries : 
- matplotlib
- seaborn
- minidom
- os
- datetime
- numpy
- pandas
- tensorflow
- scipy
- script
- IPython
- pickle
- statsmodels
- collections

## Members of group brr
- [Luca Bataillard - 282152](https://github.com/bataillard)
- [Julian Blackwell - 289803](https://github.com/JulianBlackwell)
- [Changling Li - 282440](https://github.com/lichangling3)
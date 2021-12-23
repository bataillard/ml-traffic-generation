# CS-433 - Machine Learning project 2

## Introduction
The aim of this project is to learn to use the concepts of machine learning presented in the lectures and practiced in the labs on a
real-world dataset. For this project we chose to collaborate with an EPFL lab TRANSP-OR who provided a historical traffic dataset of a bridge in Switzerland, in order to generate discrete traffic data using ML methods.

## Organisation
This project is organized as follows :

- the repository **_data_** that contains a small dataset **_extract.txt_**
- the repository **_report_** that contains the LaTeX template of our final written report.
- the repository **_src_** that includes: 
    - **_0-sumo.ipynb_** that briefly explains how to simulate traffic using "Simulation of Urban MObility" (SUMO) of the extract.txt dataset.
    - **_1-data-exploration.ipynb_** were the data exploration of our datasat is made.
    - **_2-forecasting-model-selection.ipynb_** that selects a model to predict the hourly number of cars per week.
    - **_3-sampling-interval-selection.ipynb_** that finds the most appropriate sampling interval for which to predict the number of cars.
    - **_4-rate-prediction.ipynb_** that predicts the number of vehicules, speed and weight per hour that we called "rate".
    - **_5-discrete-event-generation.ipynb_** that converts the previous rate into discrete events.
    - **_utils.py_** that contains the pipeline and helper functions.
    - several .xml files that helps to define the road to generate the traffic on SUMO.
    - the repository **_sumo-files_** that contains all the files used to setup the simulation. You can find more infos on each file in the notebook 0-sumo.ipynb 
- the file **_ML-Project-2.pdf_** which is our report that provides a full explanation of our ML system and our findings.

## How to use our project
- Just make sure to have the libraries mentioned below installed on your environment before running the cells in the jupyter notebook.
- To reproduce our setup, please run the notebooks in a successive way (from 1 to 5).
- Don't forget to put the dataset in the repository "data" at the same level of the repository "src". You can find our dataset named "405.txt" on this [link](https://drive.switch.ch/index.php/s/190lRT2jVT5bCgJ).
- To run SUMO, open XQuartz (if you use MacOS), go to repository **_src_** and type "sumo-gui -c sumo-files/hello.sumocfg" in the terminal.

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
- [sumo](https://sumo.dlr.de/docs/Downloads.php)
- [XQuartz](https://www.xquartz.org/) if using OS X

## Members of group brr
- [Luca Bataillard - 282152](https://github.com/bataillard)
- [Julian Blackwell - 289803](https://github.com/JulianBlackwell)
- [Changling Li - 282440](https://github.com/lichangling3)
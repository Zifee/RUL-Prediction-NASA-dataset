# RUL-Prediction for rotating machinery, using NASA dataset as an example.
It is a good example for the practitioner to learn using DL to predict the RUL of rotating machinery. 
This is an example that uses LSTM(based on Pytorch) to predict the remaining useful life of NASA turbofan simulation. 
Following the style of MATLAB tuition, I put all the functions in a file. 
The dataset you could download from NASA website. 
In this example, CNN-LSTM and LSTM only are used to predict the RUL.
Only FD001 is examined in this example, you can replace whatever you need in NASA dataset but please noting that the useful features need to be re-selected. 

Please cite the following reference if it is helpful for you to learning RUL prediction.
Xu, Zifei, et al. "A Novel Health Indicator for Intelligent Prediction of Rolling Bearing Remaining Useful Life based on Unsupervised Learning Model." Computers & Industrial Engineering (2023): 108999.

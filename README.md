# Neural-Network-Back-propagation
Backpropagation algorithm for neural networks using Python

dataClean.py : It is used to pre-process the raw data set. Any rows with missing attributes in data
set would be removed. For each column we perform standardization if value is numeric. If value is
categorical, it would be converted to numerical values. For output or predicted variables we convert
it numerical values. The arguments for this script are file paths of input raw data and output file
path.

neuralNets.Py : In this program we implement back propagation algorithm using the pre-processed
data from previous script (i.e. dataClean.py). This script accepts five arguments.

Steps to execute program:
For the data pre-processing, run dataClean.py with the following two parameters:
1. input file path
2. output file path

Example: 
car.txt car_cleaned.csv


To train the neural network, and see the results, execute neuralNets.py, with the following parameters:
1. Cleaned data filepath
2. Percentage of data to be used for training
3. Maximum number of Iterations
4. Number of Hidden layers
5. Number of units in each hidden layer, separated by a space

Example:
 car_cleaned.csv 90 100 2 4 2


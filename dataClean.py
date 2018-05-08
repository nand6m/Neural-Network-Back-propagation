# Machine Learning: CS6375 : Assignment-3 : Fall 2017 
# Author: Nandish
import pandas as pd
import numpy as np
import sys

def Clean_Data(df,col,colNum):
	uniqueVals = col.unique()
	newcol = col
	attrValues = {}
	j = 0
	for val in uniqueVals:
		attrValues[val] = j
		j = j+1
	new = []
	for index,row in col.iteritems():
		new.insert(index,attrValues[row])
	df[colNum] = new
	newcol = (df[colNum] - df[colNum].mean())/df[colNum].std()
	df[colNum] = newcol
	return

def last_Column(df, col,colNum):
	if col.dtype == np.int64 or col.dtype == np.float64:
		newcol = (col >= col.mean())*1
		df[colNum] = newcol
	else:
		uniqueVals = col.unique()
		newcol = col
		attrValues = {}
		j = 0
		for val in uniqueVals:
			attrValues[val] = j
			j = j+1
		new = []
		for index,row in col.iteritems():
			new.insert(index,attrValues[row])
		df[colNum] = new
		newcol = (df[colNum] - df[colNum].min())/(df[colNum].max()-df[colNum].min())
		df[colNum] = newcol
	return

inputFile = sys.argv[1]
outputFile = sys.argv[2]

data = pd.read_table(inputFile,sep='\t|,|:|\s+',index_col = False,header=None, engine = 'python')
newDf = pd.DataFrame()
data.replace(to_replace="[?]",value=np.nan,regex=True,inplace=True)
data = data.dropna()

for col in range(0,len(data.columns)-2):
	if data[col].dtype == np.int64 or data[col].dtype == np.float64:
		newcol = ((data[col]-data[col].mean())/data[col].std())
		newDf[col]= newcol
	else:
		
		Clean_Data(newDf,data[col],col)

last_Column(newDf,data[len(data.columns)-1],len(data.columns)-1)

newDf.to_csv(outputFile,sep=',',index = False)
print("\nInput raw data set used : " + sys.argv[1])
print("Processed data set generated : " + sys.argv[2])

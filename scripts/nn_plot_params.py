import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import seaborn as sns



def get_training_loss_vis(history):
	plt.plot(history.history['loss'][80:120])
	plt.plot(history.history['val_loss'][80:120])
	plt.title('Model Loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train','CV'],loc='upper right')
	plt.savefig('train_cv_loss_zoomed.png')



def get_model(X,y,neurons1=15,neurons2=12,batch_size=64):
	model = Sequential()
	model.add(Dense(neurons1,input_dim=30,kernel_initializer='normal',activation='relu'))
	model.add(Dense(neurons2,kernel_initializer='normal',activation='relu'))
	model.add(Dense(1,kernel_initializer='normal'))

	model.compile(loss='mean_squared_error',optimizer='adam',metrics=['acc'])
	history = model.fit(X,y,validation_split=0.33,epochs=150,batch_size=batch_size,verbose=10)
	return history


def split_into_training_and_test(X,y):
	X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.50,random_state=42)
	return (X_train,X_test,y_train,y_test)


def preprocessing(df_feat,df_target):
	# Join the features and the target data frames
	df = df_feat.join(df_target,lsuffix='_feat',rsuffix='_target')
	# Drop redundant column
	df.drop(['jobId_target'],axis=1,inplace=True)
	# Rename the column
	df.columns = [u'jobId', u'companyId', u'jobType', u'degree', u'major',
	       u'industry', u'yearsExperience', u'milesFromMetropolis', u'salary']

	# Assign 'category' type to the nominal/ordinal variables
	df['jobType'] = df['jobType'].astype('category')
	df['major'] = df['major'].astype('category')
	df['degree'] = df['degree'].astype('category')
	df['industry'] = df['industry'].astype('category')

	# Convert the categorical varibles into numerical codes
	df['degreeCoded'] = df['degree'].cat.codes
	df['majorCoded'] = df['major'].cat.codes
	df['industryCoded'] = df['industry'].cat.codes
	df['jobTypeCoded'] = df['jobType'].cat.codes

	# Choosing 'jobType','degree','major','industry','yearsExperience','milesFromMetropolis' as features for prediction.
	# salary is our target variable.
	selected_columns = ['jobType','degree','major','industry','yearsExperience','milesFromMetropolis','salary']
	df = df[selected_columns]
	
	# Encode the categorical variables
	df = pd.get_dummies(df)

	y = df['salary'].as_matrix()
	df.drop(['salary'],inplace=True,axis=1)
	data = df.values
	X = data[:,:-1]

	# scaler = StandardScaler()
	# X = scaler.fit_transform(X)

	return (X,y)


def read_data():
	# read the data into pandas data frames
	df_feat = pd.read_csv('train_features_2013-03-07.csv',sep=',')
	df_target = pd.read_csv('train_salaries_2013-03-07.csv',sep=',')

	return (df_feat,df_target)



if __name__ == '__main__':
	seed = 7
	np.random.seed(seed)

	df_feat,df_target = read_data()
	X,y = preprocessing(df_feat,df_target)
	X_train,X_test,y_train,y_test = split_into_training_and_test(X,y)
	history = get_model(X_train,y_train)
	# print history.history
	get_training_loss_vis(history)








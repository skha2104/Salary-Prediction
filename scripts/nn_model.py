import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from keras.optimizers import SGD
from keras.layers import Dropout


############## 2 hidden layer architecture with Best parameters #########################
############## neurons1 = number of neurons in layer 1 #############################
############## neurons2 = number of neurons in layer 2 #############################
def deeper_model(neurons1=25,neurons2=10,dropout=0.0):
	model = Sequential()
	model.add(Dense(neurons1,input_dim=30,kernel_initializer='normal',activation='relu'))
	model.add(Dense(neurons2,kernel_initializer='normal',activation='relu'))
	model.add(Dropout(dropout))
	model.add(Dense(1,kernel_initializer='normal'))

	model.compile(loss='mean_squared_error',optimizer='adam')
	return model

############## 1 hidden layer architecture with Best parameters #########################
############## neurons1 = number of neurons in layer 1 #############################
def baseline_model(neurons=25):
	model = Sequential()
	model.add(Dense(neurons,input_dim=30,kernel_initializer='normal',activation='relu'))
	model.add(Dense(1,kernel_initializer='normal'))

	model.compile(loss='mean_squared_error',optimizer='adam')
	return model


############# Perform Grid Search on the parameters ############################
def do_grid_search(model):
	############ Experimenting with the batch_size, epochs, dropout, ###########
	############ neurons in layer 1 and neurons in layer 2 #####################
	
	batch_size = [64] ###### best batch size
	epochs = [95] ######## best number of epochs
	dropout = [0.0]
	# neurons1 = [1,5,10,15,20,25,30]
	neurons1 = [15] ####### best number of neurons in layer1
	neurons2 = [12] ####### best is 10 with dropout 0.0
	# param_grid_bm = dict(epochs=epochs,batch_size=batch_size,neurons=neurons1)
	param_grid_dm = dict(epochs=epochs,batch_size=batch_size,neurons1=neurons1,neurons2=neurons2,dropout=dropout)
	grid = GridSearchCV(estimator=model,param_grid=param_grid_dm,verbose=10)
	grid_result = grid.fit(X_train,y_train)

	return (grid,grid_result)


def build_model():
	# model = KerasRegressor(build_fn=baseline_model,verbose=0)
	model = KerasRegressor(build_fn=deeper_model,verbose=0)
	return model


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
	X_train,y_train = preprocessing(df_feat,df_target)
	X_train,X_test,y_train,y_test = split_into_training_and_test(X_train,y_train)
	model = build_model()
	grid,grid_result = do_grid_search(model)

	########## Predicting and printing the results ############
	y_pred = grid.best_estimator_.predict(X_test)
	print "mean_squared_error: ", mean_squared_error(y_test,y_pred)
	print "R2 score: ", r2_score(y_test,y_pred)
	
	########## Printing the results of the grid search ########
	print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
	means = grid_result.cv_results_['mean_test_score']
	stds = grid_result.cv_results_['std_test_score']
	params = grid_result.cv_results_['params']
	for mean, stdev, param in zip(means, stds, params):
		print("%f (%f) with: %r" % (mean, stdev, param))











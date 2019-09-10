import os, sys, shutil, time
import pickle
from flask import Flask, request, jsonify, render_template,send_from_directory
import pandas as pd
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import urllib.request
import json
from geopy.geocoders import Nominatim
from flask_mail import Mail,Message
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

app.config['MAIL_SERVER']='smtp.gmail.com'
app.config['MAIL_PORT'] = 465
app.config['MAIL_USERNAME'] = 'anantisdumb@gmail.com'
app.config['MAIL_PASSWORD'] = '1fuckanant'
app.config['MAIL_USE_TLS'] = False
app.config['MAIL_USE_SSL'] = True

mail=Mail(app)



def func(X_res):

	import numpy as np # linear algebra
	import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
	import matplotlib.pyplot as plt
	import seaborn as sns
	from scipy import stats
	from scipy.stats import randint
	# prep
	from sklearn.model_selection import train_test_split
	from sklearn import preprocessing
	from sklearn.datasets import make_classification
	from sklearn.preprocessing import binarize, LabelEncoder, MinMaxScaler
	# models
	from sklearn.linear_model import LogisticRegression
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
	# Validation libraries
	from sklearn import metrics
	from sklearn.metrics import accuracy_score, mean_squared_error, precision_recall_curve
	from sklearn.model_selection import cross_val_score
	#Neural Network
	from sklearn.neural_network import MLPClassifier
	from sklearn.model_selection import learning_curve
	from sklearn.model_selection import GridSearchCV
	from sklearn.model_selection import RandomizedSearchCV
	#Bagging
	from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
	from sklearn.neighbors import KNeighborsClassifier
	#Naive bayes
	from sklearn.naive_bayes import GaussianNB 
	#Stacking
	
	from sklearn.preprocessing import LabelEncoder
	#reading in CSV's from a file path
	train_df = pd.read_csv('trainms.csv')


	#missing data
	total = train_df.isnull().sum().sort_values(ascending=False)
	percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
	missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
	#missing_data.head(20)
	#print(missing_data)


	train_df = train_df.drop(['comments'], axis= 1)
	train_df = train_df.drop(['state'], axis= 1)
	train_df = train_df.drop(['Timestamp'], axis= 1)

	

	train_df.isnull().sum().max() #just checking that there's no missing data missing...
	#train_df.head(5)

	# Assign default values for each data type
	defaultInt = 0
	defaultString = 'NaN'
	defaultFloat = 0.0

	# Create lists by data tpe
	intFeatures = ['Age']
	stringFeatures = ['Gender', 'Country', 'self_employed', 'family_history', 'treatment', 'work_interfere',
					 'no_employees', 'remote_work', 'tech_company', 'anonymity', 'leave', 'mental_health_consequence',
					 'phys_health_consequence', 'coworkers', 'supervisor', 'mental_health_interview', 'phys_health_interview',
					 'mental_vs_physical', 'obs_consequence', 'benefits', 'care_options', 'wellness_program',
					 'seek_help']
	floatFeatures = []

	# Clean the NaN's
	for feature in train_df:
		if feature in intFeatures:
			train_df[feature] = train_df[feature].fillna(defaultInt)
		elif feature in stringFeatures:
			train_df[feature] = train_df[feature].fillna(defaultString)
		elif feature in floatFeatures:
			train_df[feature] = train_df[feature].fillna(defaultFloat)
		# else:
		#     #print('Error: Feature %s not recognized.' % feature)
			
			
	###########################################
	#clean 'Gender'
	#Slower case all columm's elements
	gender = train_df['Gender'].str.lower()
	#print(gender)

	#Select unique elements
	gender = train_df['Gender'].unique()

	#Made gender groups
	male_str = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
	trans_str = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
	female_str = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

	for (row, col) in train_df.iterrows():

		if str.lower(col.Gender) in male_str:
			train_df['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

		if str.lower(col.Gender) in female_str:
			train_df['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

		if str.lower(col.Gender) in trans_str:
			train_df['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

	#Get rid of bullshit
	stk_list = ['A little about you', 'p']
	train_df = train_df[~train_df['Gender'].isin(stk_list)]


	

	###############################################




	#complete missing age with mean
	train_df['Age'].fillna(train_df['Age'].median(), inplace = True)

	# Fill with media() values < 18 and > 120
	s = pd.Series(train_df['Age'])
	s[s<18] = train_df['Age'].median()
	train_df['Age'] = s
	s = pd.Series(train_df['Age'])
	s[s>120] = train_df['Age'].median()
	train_df['Age'] = s

	#Ranges of Age
	train_df['age_range'] = pd.cut(train_df['Age'], [0,20,30,65,100], labels=["0-20", "21-30", "31-65", "66-100"], include_lowest=True)

	


	




	train_df = train_df.drop(['Country'], axis= 1)


	

	########################################################
	#missing data
	total = train_df.isnull().sum().sort_values(ascending=False)
	percent = (train_df.isnull().sum()/train_df.isnull().count()).sort_values(ascending=False)
	missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
	#missing_data.head(20)
	#print(missing_data)



	######################################################


	# # Scaling Age
	# scaler = MinMaxScaler()
	# train_df['Age'] = scaler.fit(train_df[['Age']])
	# train_df['Age'] = scaler.transform(train_df[['Age']])
	# X_res['Age']= scaler.transform(X_res[['Age']])
	# Scaling Age
	scaler = MinMaxScaler()
	train_df['Age'] = scaler.fit_transform(train_df[['Age']])
	X_res['Age']=scaler.transform(X_res[['Age']])


	

	###################################################3

	# define X and y
	feature_cols = ['Age', 'Gender', 'family_history', 'benefits', 'care_options', 'anonymity', 'leave', 'work_interfere']
	X = train_df[feature_cols]
	y = train_df.treatment


	



	# split X and y into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)
	le=LabelEncoder()
	# Iterating over all the common columns in train and test
	for col in X_test.columns.values:
	   # Encoding only categorical variables
		if X_test[col].dtypes=='object':
	   # Using whole data to form an exhaustive list of levels
			data=X_train[col].append(X_test[col])
			le.fit(data.values)
			X_train[col]=le.transform(X_train[col])
			X_test[col]=le.transform(X_test[col])
			X_res[col]=le.transform(X_res[col])





	def tuningRandomizedSearchCV(model, param_dist):
		#Searching multiple parameters simultaneously
		# n_iter controls the number of searches
		rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10, random_state=5)
		rand.fit(X, y)
		#rand.grid_scores_
		
		# examine the best model
		#print('Rand. Best Score: ', rand.best_score_)
		#print('Rand. Best Params: ', rand.best_params_)
		
		# run RandomizedSearchCV 20 times (with n_iter=10) and record the best score
		best_scores = []
		for _ in range(20):
			rand = RandomizedSearchCV(model, param_dist, cv=10, scoring='accuracy', n_iter=10)
			rand.fit(X, y)
			best_scores.append(round(rand.best_score_, 3))
		#print(best_scores)


	#################################################def treeClassifier():
	def treeClassifier():   # Calculating the best parameters
		tree = DecisionTreeClassifier()
		featuresSize = feature_cols.__len__()
		param_dist = {"max_depth": [3, None],
				  "max_features": randint(1, featuresSize),
				  "min_samples_split": randint(2, 9),
				  "min_samples_leaf": randint(1, 9),
				  "criterion": ["gini", "entropy"]}
		#tuningRandomizedSearchCV(tree, param_dist)
		
		# train a decision tree model on the training set
		tree = DecisionTreeClassifier(max_depth=3, min_samples_split=8, max_features=6, criterion='entropy', min_samples_leaf=7)
		tree.fit(X_train, y_train)
		
		#make class predictions for the testing set
		y_pred = tree.predict(X_res)
		print(y_pred)
		return y_pred
		#print('########### Tree classifier ###############')
		
		#accuracy_score = evalClassModel(tree, y_test, y_pred_class, True)

		#Data for final graph
		#methodDict['Tree clas.'] = accuracy_score * 100

	return treeClassifier()
	


'''*************************END*******************************'''

@app.route('/')
def root():
	msg=Message('Hello',sender='anantisdumb@gmail.com',recipients=['anantisdumb@gmail.com'])
	msg.body="This is the email body"
	mail.send(msg)
	return render_template('home.html')

@app.route('/form1')
def form1():
	return render_template('form1.html')

@app.route('/form2')
def form2():
	return render_template('form2.html')

@app.route('/form3')
def form3():
	return render_template('form3.html')

@app.route('/form4')
def form4():
	return render_template('form4.html')

@app.route('/images/<Paasbaan>')
def download_file(Paasbaan):
	return send_from_directory(app.config['images'], Paasbaan)

@app.route('/index.html')
def index():
	return render_template('index.html')

@app.route('/work.html')
def work():
	return render_template('work.html')

@app.route('/about.html')
def about():
	return render_template('about.html')

@app.route('/contact.html')
def contact():
	return render_template('contact.html')

@app.route('/result.html', methods = ['POST'])
def predict():
	if request.method == 'POST':
		age=request.form['age']
		gender=request.form['gender']
		history=request.form['History']
		benefits=request.form['seek_help']
		care_opt=request.form['care_options']
		anonymity=request.form['anonymity']
		leave=request.form['leave']
		work=request.form['intf']

		# print(age)
		# print(gender)
		# print(history)
		# print(benefits)
		# print(care_opt)
		# print(anonymity)
		# print(leave)
		# print(work)
		
		X_res= pd.DataFrame({"Age": [age],
				"Gender":[gender],
				"family_history":[history],
				"benefits":[benefits],
				"care_options":[care_opt],
				"anonymity":[anonymity],
				"leave":[leave],
				"work_interfere":[work]})
		
		# print(X_res)

		
		
		
		y_res=func(X_res)
			 

		print("**********************************")
		print("**********************************")
		print(y_res)
		print("**********************************")
		print("**********************************")
	
		
		if y_res == "Yes":
			my_prediction='Treatment is required.Sanction Leave'
			#print("ROBBERY")
		else:
			my_prediction='Treatment not Needed'
			#print("SAFE")
		
		

	return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
	app.run(debug = True)

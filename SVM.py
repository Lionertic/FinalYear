import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.feature_selection import RFE
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
from sklearn.externals.six import StringIO  



def process(path):
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
       

	# Load the digits dataset
	df = pd.read_csv(path)
	df.columns = ['having_IP_Address',
	'URL_Length',
	'Shortining_Service',
	'having_At_Symbol',
	'double_slash_redirecting',
	'Prefix_Suffix',
	'having_Sub_Domain',
	'SSLfinal_State',
	'Domain_registeration_length',
	'Favicon',
	'port',
		'HTTPS_token',
	'Request_URL',
	'URL_of_Anchor',
	'Links_in_tags',
	'SFH',
	'Submitting_to_email',
	'Abnormal_URL',
		'Redirect',
	'on_mouseover',
	'RightClick',
	'popUpWidnow',
	'Iframe',
	'age_of_domain',
	'DNSRecord',
	'web_traffic ',
	'Page_Rank',
	'Google_Index',
	'Links_pointing_to_page',
	'Statistical_report',
	'Result']

	X = df[df.columns[:-1]].values
	y = df[['Result']].values
	names = df.head()

	# Create the RFE object and rank each pixel
	svc = SVC(kernel="linear", C=1)
	rfe = RFE(estimator=svc, n_features_to_select=3, step=1)
	rfe.fit(X, y)
	# summarize the selection of the attributes
	print(rfe.support_)
	print(rfe.ranking_)
	print ("Features sorted by their rank:")
	print (sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names)))
	mm=sorted(zip(map(lambda x: round(x, 4), rfe.ranking_), names))

	a1=mm[0]
	a2=mm[1]
	a3=mm[2]


	XX=[]
	YY=[]
	XX.append(a1[1])
	YY.append(1)
	XX.append(a2[1])
	YY.append(1)
	XX.append(a3[1])
	YY.append(1)
    
	#Barplot for the dependent variable
	fig = plt.figure(0)
	plt.bar(XX,YY,align='center', alpha=0.5,color=colors)
	plt.xlabel('Feature Selection')
	plt.ylabel('RANK')
	plt.title("Feature Selection BY SVM");
	fig.savefig('results/SVMFeatureSelection.png') 	
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	cols=[]
	cols.append(a1[1])
	cols.append(a2[1])
	cols.append(a3[1])

	sns.set(style='whitegrid', context='notebook')
	sns.pairplot(df[cols], size=1.5);
	plt.pause(5)
	plt.show(block=False)
	plt.close()
	
	fig = plt.figure(0)
	cm = np.corrcoef(df[cols].values.T) 
	sns.set(font_scale = 1.5)
	hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 15}, yticklabels=cols, xticklabels=cols)
	fig.savefig('results/SVMSNS.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()

	X_new = rfe.transform(X) 
	X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size = 0.25)
	svc.fit(X_train, y_train)
	y_pred = svc.predict(X_test)

	mse=mean_squared_error(y_test, y_pred)
	mae=mean_absolute_error(y_test, y_pred)
	r2=r2_score(y_test, y_pred)


	print("MSE VALUE FOR SVM IS %f "  % mse)
	print("MAE VALUE FOR SVM IS %f "  % mae)
	print("R-SQUARED VALUE FOR SVM IS %f "  % r2)
	rms = np.sqrt(mean_squared_error(y_test, y_pred))
	print("RMSE VALUE FOR SVM IS %f "  % rms)
	ac=accuracy_score(y_test, y_pred)
	print ("ACCURACY VALUE SVM IS %f" % ac)
	print("------------------------------------------------------------------")	
	result2=open("results/resultSVM.csv","w")
	result2.write("ID,Predicted Value" + "\n")
	for j in range(len(y_pred)):
	    result2.write(str(j+1) + "," + str(y_pred[j]) + "\n")
	result2.close()

	result2=open('results/SVMMetrics.csv', 'w')
	result2.write("Parameter,Value" + "\n")
	result2.write("MSE" + "," +str(mse) + "\n")
	result2.write("MAE" + "," +str(mae) + "\n")
	result2.write("R-SQUARED" + "," +str(r2) + "\n")
	result2.write("RMSE" + "," +str(rms) + "\n")
	result2.write("ACCURACY" + "," +str(ac) + "\n")
	result2.close()
	
	
	df =  pd.read_csv('results/SVMMetrics.csv')
	acc = df["Value"]
	alc = df["Parameter"]
	colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#8c564b"]
	explode = (0.1, 0, 0, 0, 0)  
	
	fig = plt.figure()
	plt.bar(alc, acc,color=colors)
	plt.xlabel('Parameter')
	plt.ylabel('Value')
	plt.title(' SVM Metrics Value')
	fig.savefig('results/SVMMetricsValue.png') 
	plt.pause(5)
	plt.show(block=False)
	plt.close()


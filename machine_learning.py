import numpy as np
from sklearn import linear_model, metrics, decomposition, datasets
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from random import sample
from math import floor
import seaborn as sns
from sklearn.decomposition import PCA
import pandas as pd
from regressors import stats


def pca(data, components):
	""" Plot variances explaines by PC
	"""

	_pca = PCA(n_components = components)
	_pca.fit(data)
	var = _pca.explained_variance_ratio_
	cum_var = np.cumsum(np.round(var, decimals=4)*100)
	fig = plt.plot(cum_var)
	rotation = pd.DataFrame(
		_pca.components_,
		columns = data.columns,
		index = ['PC-1','PC-2','PC-3','PC-4','PC-5','PC-6','PC-7','PC-8','PC-9',]
		)

	return (fig, rotation)



def LinearRegression(variables, target):
	"""Ridge regression
	Type variables & target = dataframe
	Attention target & variables doivent avoir les mêmes index
	"""

	index_test = sample(list(variables.index), floor(len(variables.index) * 0.8))
	index_validation = [i for i in list(variables.index) if i not in index_test]

	variables_test = variables.loc[index_test]
	variables_validation = variables.loc[index_validation]

	target_test = target.loc[index_test]
	target_validation = target.loc[index_validation]

	linear_ridge = linear_model.Ridge()
	C_s = np.logspace(-10,10,10)
	parameters = {'alpha' : C_s}
	clf = GridSearchCV(linear_ridge, parameters)
	clf.fit(variables_test.values, target_test.values)
	best_estimator = clf.best_params_["alpha"]

	test_predicted = clf.predict(variables_test.values)
	target_predicted = clf.predict(variables_validation)

	best_clf = linear_model.Ridge(alpha = best_estimator)
	best_clf.fit(variables_test.values, target_test.values)
#	stats.coef_pval(best_clf, variables_test.values, target_test.values)
	coefs = pd.DataFrame.from_dict({'index' : list(variables_test), 'coefs' : list(best_clf.coef_[0])}).set_index('index')

	return {
#	'result' : clf.cv_results,
	'r2_score_validation' : metrics.r2_score(target_predicted, target_validation.values),
	'model_espace_test' : pd.DataFrame({'index' : index_test, 
										'real' : list(target_test.values), 
										'model' : list(test_predicted)}).set_index('index'),
	'model_espace_validation' : pd.DataFrame({'index' : index_validation,
												'real' : list(target_validation.values),
												'predicted' : list(target_predicted)}).set_index('index'),
	'coefs' : coefs
		}


def LogisticRegression(variables, targeted):
	""" Classification report for the Pipeline PCA + Logistic"""

	logistic = linear_model.LogisticRegression()
	pca = decomposition.PCA()
	pipe = Pipeline(steps=[('pca', pca), ('logistic', logistic)])

	n_components = [4,5]
	Cs = np.logspace(-10,10,10)

	estimator = GridSearchCV(
		pipe,
		dict(pca__n_components = n_components,
			logistic__C = Cs)
		)
	
	predicted = cross_val_predict(
	estimator, variables, targeted, cv = 5)

	return {
		'accuracy':metrics.accuracy_score(targeted, predicted),
		'report':metrics.classification_report(targeted, predicted)
		}


def cross_val_SVM_linear(variables, targeted):
	""" Classification report for the Pipeline PCA + SVM Linear"""

	svm = SVC(kernel = "linear")
	pca = decomposition.PCA()
	pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])

	n_components = [4,5]
	C = np.logspace(-10,10,10)

	estimator = GridSearchCV(
		pipe,
		dict(pca__n_components = n_components,
			svm__C = C)
		)
	
	predicted = cross_val_predict(
	estimator, variables, targeted, cv = 5)

	return {
		'accuracy':metrics.accuracy_score(targeted, predicted),
		'report':metrics.classification_report(targeted, predicted)
		}

def cross_val_SVM_RBF(variables, targeted):
	""" Classification report for the Pipeline PCA + SVM RBF"""

	svm = SVC(kernel = "rbf")
	pca = decomposition.PCA()
	pipe = Pipeline(steps=[('pca', pca), ('svm', svm)])

	n_components = [4,5]
	C = np.logspace(-10,10,10)

	estimator = GridSearchCV(
		pipe,
		dict(pca__n_components = n_components,
			svm__C = C)
		)
	
	predicted = cross_val_predict(
	estimator, variables, targeted, cv = 5)

	return {
		'accuracy':metrics.accuracy_score(targeted, predicted),
		'report':metrics.classification_report(targeted, predicted)
		}

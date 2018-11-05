import timeit
import os
import numpy as np
from helpers import *
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import accuracy_score


def create_ann(name):
    """ Construct the multi-layer perceptron classifier object for a given
    dataset based on A1 best parameters.

    Args:
        name (str): Name of dataset.
    Returns:
        ann (sklearn.MLPClassifier): Neural network classifier.

    """
    ann = MLPClassifier(activation='relu', max_iter=5000,
                        solver='adam', learning_rate='adaptive')

    if name == 'Visa':
        hls = (105, 5, 5)
        alpha = 0.001
        ann.set_params(hidden_layer_sizes=hls, alpha=alpha)
    elif name == 'Letter':
        hls = (105, 5, 5)
        alpha = 0.001
        ann.set_params(hidden_layer_sizes=hls, alpha=alpha)

    return ann


def ann_experiment(X, y, name, ann):
    """Run ANN experiment and generate accuracy and timing score

    Args:
        X (Numpy.Array): Attributes.
        y (Numpy.Array): Labels.

    """
    # get training and test splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=0, stratify=y)

    # train model
    start_time = timeit.default_timer()
    ann.fit(X_train, y_train)
    end_time = timeit.default_timer()
    elapsed = end_time - start_time

    # get predicted labels using test data and score
    y_pred = ann.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return acc, elapsed

def main():
	"""Run code to generate results.

	"""
	combined = './results/combined_results.csv'

	try:
		os.remove(combined)
	except:
		pass

	with open(combined, 'a') as f:
		f.write('dataset,algorithm,accuracy,elapsed_time\n')

	names = ['Visa']
	dimred_algos = ['PCA', 'ICA', 'RP', 'RF']
	cluster_algos = ['km', 'gmm']

    # generate results
	for name in names:
		# get labels
		X, y, letter_x, letter_y = load_data()
        
		ann = create_ann(name=name)
		acc, elapsed = ann_experiment(X, y, name, ann)
        
		with open(combined, 'a') as f:
			f.write('{},{},{},{}\n'.format(name, 'base', acc, elapsed))

		for d in dimred_algos:
			# get attributes
			X = np.loadtxt('./results/Visa_{}_projected.csv'.format(d), delimiter=',')

			# train ANN and get test score, elapsed time
			ann = create_ann(name=name)
			acc, elapsed = ann_experiment(X, y, name, ann)
			with open(combined, 'a') as f:
				f.write('{},{},{},{}\n'.format(name, d, acc, elapsed))

		for c in cluster_algos:
			# get attributes
			X = np.loadtxt('./results/Visa_{}_labels.csv'.format(c), delimiter=',')

			# train ANN and get test score, elapsed time
			ann = create_ann(name=name)
			acc, elapsed = ann_experiment(X, y, name, ann)
			with open(combined, 'a') as f:
				f.write('{},{},{},{}\n'.format(name, c, acc, elapsed))


if __name__ == '__main__':
	main()
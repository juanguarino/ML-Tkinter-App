"""
The utility module:
Manipulate data included in Sklearn datasets for training, test and evaluate machine learning methods.

Include functions to:
- Calculate distances between two points
- Generate random centers
- Find nearest point from a point
- Calculate new centers for clustering
- Calculate distances from centers and new centers
- Display datasets using tkinter
"""

# Import modules
import tkinter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn import metrics
import matplotlib
from matplotlib.figure import Figure
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Dictionary with classifiers
methods_dict = {
        'K-nearest neighbours':KNeighborsClassifier(),
        'Support Vector Machine':SVC(kernel='rbf')
}

# Dictionary with datasets
datasets_dict = {
    'Iris dataset':datasets.load_iris(),
    'Breast Cancer dataset':datasets.load_breast_cancer(),
    'Wine Quality dataset':datasets.load_wine()
}

# Dictionary with parameters
parameters = {
        'K-nearest neighbours':[{'n_neighbors': range(1,30)},'K Nearest Neighbors'],
        'Support Vector Machine':[{'gamma': [0.0001, 0.001, 0.01, 0.1, 1.0]}, 'Gamma']
}

# Function for modeling
def modeling(method,dataset,canvas, k):
        '''
        Modeling and plotting the selected dataset, using a chosen method and k value.

        :param method: Selected machine learning classifier
        :param dataset: Selected datasets from sklearn datasets
        :param canvas: Canvas to display plots
        :param k: number of K-fold for cross validation
        :return: text for displaying on canvas
        '''

        # Read dataset dictionary
        data = datasets_dict[dataset]

        # Create predictor and label variable
        X = data.data
        y = data.target

        # Set class names
        class_names = data.target_names

        # Split dataset into training set and testing set
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

        # Get classifier
        classifier = methods_dict[method]

        # Get parameters for hyperparameter tuning
        parameter = parameters[method]

        # Fit cross validation function for hyperparameter tuning
        gscf_classifier = GridSearchCV(estimator=classifier,
                                       param_grid=parameter[0],
                                       cv=KFold(k),
                                       scoring='accuracy')

        # Fit selected classifier
        gscf_classifier.fit(X_train, y_train)

        # Cross validation means
        means = gscf_classifier.cv_results_['mean_test_score']

        # Label prediction on testing set
        y_pred = gscf_classifier.predict(X_test)

        # Accuracy score on testing set
        accuracy = metrics.accuracy_score(y_test, y_pred) * 100

        # X axis for plotting
        x = list(parameter[0].values())[0]
        # Y axis for plotting
        y = means

        # Plot Score vs Parameter
        figure1 = Figure(figsize=(5, 4), dpi=100)
        subplot1 = figure1.add_subplot(111)
        subplot1.plot(x, y, color='purple', marker='o', markeredgecolor='black')
        subplot1.set(xlabel=parameter[1], ylabel='Score', title='Score Vs Parameter')
        plot1 = FigureCanvasTkAgg(figure1, canvas)
        plot1.get_tk_widget().pack(side=tkinter.BOTTOM, fill=tkinter.BOTH, expand=1)

        # Plot confusion matrix
        figure2 = Figure(figsize=(5.5, 4), dpi=100)
        subplot2 = figure2.add_subplot(111)
        plot_cmatrix = metrics.plot_confusion_matrix(gscf_classifier, X_test, y_test, ax=subplot2, display_labels=class_names)
        plot_cmatrix.ax_.set_title('Accuracy = {0:.2f}%'.format(accuracy))
        plot2 = FigureCanvasTkAgg(figure2, canvas)
        plot2.get_tk_widget().pack(expand=1)

        # Get best parameter for displaying in Text box
        text = print_best_params(gscf_classifier)

        return text

def print_best_params(classifier):
        '''
        Create a text with best parameters and accuracy means for each parameter tested

        :param classifier: classifier fitted with best parameters
        :return: text with scores and parameters
        '''

        # Get cross validation results
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        params = classifier.cv_results_['params']

        # Create text to display
        text = f'Best parameters : {classifier.best_params_}\n\n'

        # Iterate over parameters and results and add to text
        for mean, std, param in zip(means, stds, params):
                text += f'Parameter: {param}, accuracy: {mean:0.3f} (+/-{std:0.03f})\n'

        return text

import datetime

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

PATH = r'/Users/julien/Desktop/ESILV/A3/Dorset/Python/Data analysis - Assignement 2/data/heart_disease_synthetic_dataset.csv'

df = pd.read_csv(PATH)

X = df.drop(columns='target')
Y = df['target']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, train_size=0.7, random_state=42)

def run_model(model, model_name, preprocessing = None, saveResult = True):
    """
    Train a given scikit-learn model on the training data, evaluate its performance on the test data,
    and generate a confusion matrix plot.

    :param model: The scikit-learn machine learning model to train and evaluate. This model should
                  implement the `fit` and `predict` methods as scikit-learn models do.
    :param model_name: A string representing the name of the model. This is used for
                       identifying the model in the output and saving the confusion matrix plot.
    :param preprocessing: Specifies the type of preprocessing to apply to the data before training.
                          If set to 'rescaled', the features will be scaled using MinMaxScaler.
                          Else, the original features without scaling are used.

    This function performs the following steps:
    1. Depending on the `preprocessing` parameter, either rescales the features using MinMaxScaler
       or uses the original features without scaling.
    2. Trains the model using the training data (X_train, Y_train) with the `fit` method.
    3. Predicts the labels for the test data (X_test) using the `predict` method.
    4. Calculates and prints the accuracy, precision, recall, and F1 score of the model.
    5. Generates and saves a confusion matrix plot for the model's predictions.

    The confusion matrix plot is saved as an image file in the "resources" directory,
    with the filename format "{model_name}_confusion_matrix.png".
    """
    if preprocessing == 'rescaled':
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model.fit(X_train_scaled, Y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

    accuracy = accuracy_score(Y_test, y_pred)
    precision = precision_score(Y_test, y_pred)
    recall = recall_score(Y_test, y_pred)
    f1 = f1_score(Y_test, y_pred)

    print(f"+-{("-"*len(model_name))}---------+")
    print(f"| {model_name} results |")
    print(f"+-{("-"*len(model_name))}---------+")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    if saveResult:
        with open("resources/results.txt", "a") as f:
            f.write(f"{model_name}\n")
            f.write(f"{("-"*len(model_name))}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Precision: {precision:.4f}\n")
            f.write(f"Recall: {recall:.4f}\n")
            f.write(f"F1 Score: {f1:.4f}\n\n")

    cm = confusion_matrix(Y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.savefig(f"resources/{model_name}_confusion_matrix.png")
    plt.close()

with open("resources/results.txt", "a") as f:
    f.write(f"+--------------------------------------+\n")
    f.write(f"| Test from {datetime.datetime.now()} |\n")
    f.write(f"+--------------------------------------+\n\n")

for k in range(3, 11):
    run_model(KNeighborsClassifier(n_neighbors=k), f"KNN_{k}")

for kernel in ['linear', 'poly', 'rbf', 'sigmoid']:
    run_model(SVC(kernel=kernel), f"SVM_{kernel}")

for activation_function in ['relu', 'identity', 'logistic', 'tanh']:
    run_model(MLPClassifier(activation=activation_function), f"MLP_{activation_function}")

with open("resources/results.txt", "a") as f:
    f.write("\n+-------------+\n")
    f.write("| End of test |\n")
    f.write("+-------------+\n\n")
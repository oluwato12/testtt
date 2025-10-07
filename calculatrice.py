 #Pour les données de iris
from sklearn.datasets import load_iris

#Pour entrainer le modele et tester (train et test)
from sklearn.model_selection import train_test_split

#Pour l'algorithe d'apprentissage automatique (abre de decisison)
from sklearn.tree import DecisionTreeClassifier

#Pour le pourcenetage de bonne decision 
from sklearn.metrics import accuracy_score

#Avoir les donnees sous forme tableau 
import pandas as pd
iris = load_iris()
#Donnees des fleur
x = iris.data
#Labels
y = iris.target

x_train, x_test, y_train ,y_test = train_test_split(x, y, test_size=0.2, random_state=42)

#Creaion le modele
model = DecisionTreeClassifier()

#Entrenement du modele
model.fit(x_train,y_train)

#Pour faire des predictions 
y_pred = model.predict(x_test)

#Verifier l'exatitude du modele 
accuracy =  accuracy_score(y_test, y_pred)

print("Taux de prédiction",accuracy)
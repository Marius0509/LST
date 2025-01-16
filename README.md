# Satelite scalers Project

## The idea
Proiectul consta in utilizarea inteligentei artificiale pentru a creste rezolutia imaginilor satelitare. Pentru asta, am dezolvoltat 2 modele.

## Graph Convolutional Network / Graph Neural Network
Dupa cum se cerea si in cerinta, am inceput prin dezvoltarea unui model de tip GCN pentru a face acest task.  Acest model are un input de tip (value, posx, posy) si output de forma (9,9) pentru rezultat. Graful pentru fiecare pixel se forma impreuna cu toti cei 8 vecini ai sai.<br>
Modelul a fost antrenat pe poze din data setul Landsat 8 Collection 2 Level 2 Band 10.  Aceasta ruta a avut unele limitari, iar de aceea am dezvoltat un alt tip de model, chiar daca modelul in sine este functional.

## Convolutional Neural Network 
Am dezvoltat un model Convolutional Neural Network (CNN) in keras cu input (25, 25, 1) si output (3, 3). Inputul reprezinta o zona de 25x25 pixeli, iar outputul reprezinta pixelul din centrul inputului super-rezolvat. Modelul a fost antrenat pe acelasi dataset, fiecare poza fiind impartita in zone de 75x75 pixeli, din care s-au obtinut perechile de input si output.<br>
Deoarece pentru a super-rezolva un pixel este nevoie de zona de 25x25 de pixeli care are ca centru pixelul respectiv, marginile imaginii, formate din primii 12 pixeli în fiecare directie, nu au suficienti vecini pentru a fi super-rezolvate, deci dintr-o imagine de input NxM se va obtine o imagine de output (N-24)*3x(M-24)*3.<br>

## App
Aplicatia consta intr-un client frontend care ruleaza in browser, implementat cu html, javascript si css, si un server backend dezvoltat in python care utilizeaza flask.<br><br>

## Results
Mean Squared Error (MSE): 0.13340785437655 <br>
Mean Absolute Error (MAE): 0.0013708111975406 <br>
R-squared (R2): 0.954156922595998 <br>
Pentru modelul de CNN. 

## Team members - AI
Vaida Bogdan Alexandru Darius <br>
Voina Marius Alexandru

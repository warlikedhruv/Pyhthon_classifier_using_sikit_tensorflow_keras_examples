import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing, feature_selection, ensemble, linear_model, metrics, decomposition, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans
from tensorflow.keras.layers import Input, Dense, Activation,Dropout
from tensorflow.keras.models import Model
import tensorflow as tf
import keras
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

df = pd.read_csv("training-set.csv",header=None)
df1 = df.drop(columns=[48,23])
print(df1)
scaler = StandardScaler()
df2 = pd.DataFrame(scaler.fit_transform(df1))
classes = df[48].values.tolist()
print(classes)
df2[48] = classes
print(df2)
#corr_matrix = df1.copy()
#plt.figure(figsize=(16, 6))
#heatmap = sns.heatmap(corr_matrix)
#heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
#plt.show()

df_train, df_test = model_selection.train_test_split(df2,
                      test_size=0.3)

X_train = df_train.drop(columns=[48])
y_train = df_train[48]
X_test = df_test.drop(columns=[48])
y_test = df_test[48]

X = X_train.to_numpy()
y = y_train.to_numpy()

#SVM
param_grid = {'C': [0.1, 1, 10, 100, 1000],
              'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
              'kernel': ['rbf', 'linear']}

clf = svm.SVC(kernel='rbf', C=0.1)
clf.fit(X_train,y_train)
y_predict = clf.predict(X_test)
print("Accuracy SVM :",metrics.accuracy_score(y_test, y_predict))

#random forest clasifier
RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(X_train, y_train)
y_predict2 = RF.predict(X_test)
print("Accuracy Forest classifier :",metrics.accuracy_score(y_test, y_predict2))

#neural network
NN = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
NN.fit(X_train, y_train)
y_predict3 = NN.predict(X_test)
print("Accuracy Neural network :",metrics.accuracy_score(y_test, y_predict3))

#KMeans
km = KMeans(
    n_clusters=2, init='random',
    n_init=10, max_iter=300,
    tol=1e-04, random_state=0
)
y_km = km.fit_predict(X_train)
y_predict4 = km.predict(X_test)
print("Accuracy KMeans :",metrics.accuracy_score(y_test, y_predict4))

#tensorflow Model
input_layer = Input(shape=(X.shape[1],))
dense_layer_1 = Dense(15, activation='relu')(input_layer)
dense_layer_2 = Dense(10, activation='relu')(dense_layer_1)
output = Dense(2, activation='softmax')(dense_layer_2)

model = Model(inputs=input_layer, outputs=output)
model.compile(loss="mean_squared_error" , optimizer="adam", metrics=["mean_squared_error"])
history = model.fit(X_train, y_train, batch_size=8, epochs=50, verbose=1, validation_split=0.2)
score = model.evaluate(X_test, y_test, verbose=1)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])

#keras

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(X.shape[1],)),
    keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(128, activation=tf.nn.relu),
	keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),

    keras.layers.Dense(1, activation=tf.nn.sigmoid),
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=1)
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy KERAS:', test_acc)
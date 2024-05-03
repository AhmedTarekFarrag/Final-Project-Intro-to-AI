import numpy as np
import pandas as pd
import tensorflow.keras as keras
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, Input
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Read the dataset
df = pd.read_csv('C:/Users/Public/Documents/Credittt/creditcard.csv')

# Drop the 'Time' column
df = df.drop(['Time'], axis=1)

# Drop rows with missing values
df.dropna(inplace=True)

# Separate features and labels
X = df.drop(['Class'], axis=1)
y = df['Class']

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Scaling the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Autoencoder
input_layer = Input(shape=(X_train_scaled.shape[1],))
encoder = Dense(20, activation="relu")(input_layer)
encoder = Dense(15, activation="relu")(encoder)

decoder = Dense(20, activation="relu")(encoder)
decoder = Dense(X_train_scaled.shape[1], activation='relu')(decoder)

autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# Encoder model
encoder_model = keras.Model(inputs=input_layer, outputs=encoder)

optimizer = keras.optimizers.RMSprop(learning_rate=0.1)
autoencoder.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer=optimizer)

# Training the autoencoder
history = autoencoder.fit(X_train_scaled, X_train_scaled,
                           epochs=25,
                           batch_size=32,
                           validation_data=(X_test_scaled, X_test_scaled)).history

# Encoding train and test data
X_train_encoded = encoder_model.predict(X_train_scaled)
X_test_encoded = encoder_model.predict(X_test_scaled)

# Using Logistic Regression for classification
logreg = LogisticRegression()
logreg.fit(X_train_encoded, y_train)
yhat_train = logreg.predict(X_train_encoded)
yhat_test = logreg.predict(X_test_encoded)

# Calculate accuracy
train_acc = accuracy_score(y_train, yhat_train)
test_acc = accuracy_score(y_test, yhat_test)
print("Classifier Used: Logistic Regression")
print("Train Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Print training data ratio
print("Training Data Ratio:", len(X_train) / len(df))

# Print testing data ratio
print("Testing Data Ratio:", len(X_test) / len(df))


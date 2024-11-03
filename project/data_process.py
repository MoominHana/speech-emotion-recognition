import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import tensorflow.keras as keras

class DataProcess(object):
    
    def process_data(self, df, file_dir):
        
        # Initialize lists for features and labels
        features = []
        labels = []
        genders = [] 

        # Read WAV files and extract features
        for f in df.index:
            file = file_dir + f
            try:
                signal, rate = librosa.load(file, sr=None)
                df.at[f, 'length'] = len(signal) / rate

                # Extract MFCCs
                mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)

                if mfccs_mean.shape[0] == 13:
                    features.append(mfccs_mean)
                    labels.append(df.at[f, 'Emotion'])  
                    genders.append(df.at[f, 'Gender'])  

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        # Convert features and labels to arrays
        X = np.array(features)
        y = np.array(labels)
        genders = np.array(genders)

        # Create a DataFrame
        df_features = pd.DataFrame(X)
        df_features['Emotion'] = y
        df_features['Gender'] = genders
        
        return df_features

    def split_data(self, df):
        # Prepare features and labels for the model
        X = df.drop(columns=['Emotion', 'Gender']).values
        y = df['Emotion'].values

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        return X, y, X_train, X_test, y_train, y_test
    
    def show_cmatrix(self, y_test, y_pred, y):
        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, y_pred)

        # Plotting the confusion matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()
    
#     def process_data(test_size, validation_size, df):
        
#         X, y = extrac_all_mfcc(df)
        
#         # create train, validation and test split
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
#         X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

#         # add an axis to input sets
#         X_train = X_train[..., np.newaxis]
#         X_validation = X_validation[..., np.newaxis]
#         X_test = X_test[..., np.newaxis]

#         return X_train, X_validation, X_test, y_train, y_validation, y_test
    
#     def build_model(input_shape):
#         """Generates CNN model

#         :param input_shape (tuple): Shape of input set
#         :return model: CNN model
#         """

#         # build network topology
#         model = keras.Sequential()

#         # 1st conv layer
#         model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
#         model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
#         model.add(keras.layers.BatchNormalization())

#         # 2nd conv layer
#         model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
#         model.add(keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
#         model.add(keras.layers.BatchNormalization())

#         # 3rd conv layer
#         model.add(keras.layers.Conv2D(32, (2, 2), activation='relu'))
#         model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
#         model.add(keras.layers.BatchNormalization())

#         # flatten output and feed it into dense layer
#         model.add(keras.layers.Flatten())
#         model.add(keras.layers.Dense(64, activation='relu'))
#         model.add(keras.layers.Dropout(0.3))

#         # output layer
#         model.add(keras.layers.Dense(10, activation='softmax'))

#         return model
    
    




import os
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import parselmouth

import tensorflow.keras as keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout


class DataProcess(object):
    
    def process_data(self, df, file_dir):
        
        # Initialize lists for features and labels
        features = []
        labels = []
        genders = [] 

        # Read WAV files and extract features
        for index, file_name in zip(df.index, df['File_Name']):
#             print(file_name)
            file = file_dir + file_name
            try:
                signal, rate = librosa.load(file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5) 
                df.at[index, 'length'] = len(signal) / rate

                # FEATURE1: Extract MFCCs
                mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)
                
                # Create snd object for exracting features
                snd = parselmouth.Sound(file)
                
                # FEATURE2: Extract pitch
                pitch = snd.to_pitch()
                pitch_values = pitch.selected_array['frequency']
                # Get mean Pitch
#                 print("Pitch ")
#                 print(len(pitch_values))
                pitch_mean = np.mean(pitch_values)
                
                # FEATURE3: Extract intensity
                intensity = snd.to_intensity()
                intensity_values = intensity.values.T.flatten()
                # Get Mean Intensity
#                 print("\nIntensity ")
#                 print(len(intensity_values))
                intensity_mean = np.mean(intensity_values)
                
                # FEATURE4: Extract Zero-Crossing Rate
                zcr = librosa.feature.zero_crossing_rate(y=signal)
#                 print("\nZCR ")
#                 print(len(zcr))
                zcr_mean = np.mean(zcr)
                
                # Checking
                if mfccs_mean.shape[0] == 13:
                    all_features = np.append(mfccs_mean, [zcr_mean, pitch_mean, intensity_mean])
                    features.append(all_features)
                    labels.append(df.at[index, 'Emotion'])  
                    genders.append(df.at[index, 'Gender'])  

            except Exception as e:
                print(f"Error processing file {file}: {e}")

#         print(len(labels))
        
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
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Confusion Matrix')
        plt.show()
        
    def process_data_nn(self, df, file_dir):
        
        # Initialize lists for features and labels
        features = []
        labels = []
        genders = [] 
        target_frames = 216

        # Read WAV files and extract features
        for index, file_name in zip(df.index, df['File_Name']):
            
            file = file_dir + file_name
            try:
                signal, rate = librosa.load(file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5) 
                df.at[index, 'length'] = len(signal) / rate

                # FEATURE1: Extract MFCCs
                mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)
                
                # Checking
                if mfccs.shape[0] == 13:
                
                    if mfccs.shape[1] < target_frames:
                        print("Less: ", mfccs.shape[1])
                        # Pad with zeros if fewer frames
                        mfccs = np.pad(mfccs, ((0, 0), (0, target_frames - mfccs.shape[1])), mode='constant')
                    elif mfccs.shape[1] > target_frames:
                        # Trim if more frames
                        print("More: ", mfccs.shape[1])
                        mfccs = mfccs[:, :target_frames]
                        
                    features.append(mfccs)
                    labels.append(df.at[index, 'Emotion'])  
                    genders.append(df.at[index, 'Gender'])  

            except Exception as e:
                print(f"Error processing file {file}: {e}")

        # Example data
        X = np.array(features)
        y = np.array(labels) 
        
        # Encode labels
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        
#         print("Mapping of labels to numbers:", dict(zip(label_encoder.classes_, range(len(label_encoder.classes_)))))

        class_names = list(label_encoder.classes_)
        print(class_names)

        
        return X, y, class_names
    
    def split_data_nn(self, X, y, test_size, validation_size):
        
        # Split data 
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size, random_state=42)
        
        return X_train, X_validation, X_test, y_train, y_validation, y_test
    
    def define_nn(self, X):
        # Define the model
        model = keras.Sequential([
            keras.layers.Flatten(input_shape = (X.shape[1], X.shape[2])),
            keras.layers.Dense(512, activation = "relu"),
            keras.layers.Dense(512, activation = "relu"),
            keras.layers.Dense(512, activation = "relu"),
            keras.layers.Dense(8, activation = "softmax")
        ])
        
        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model
    
    def process_data_cnn(self, df, file_dir):
        # Initialize lists for features and labels
        features = []
        labels = []
        genders = [] 

        # Read WAV files and extract features
        for f in df['File_Name']:
            file = file_dir + f
            try:
                signal, rate = librosa.load(file, sr=None)
                df.at[f, 'length'] = len(signal) / rate

                # Extract MFCCs and calculate both mean and standard deviation for each coefficient
                mfccs = librosa.feature.mfcc(y=signal, sr=rate, n_mfcc=13)
                mfccs_mean = np.mean(mfccs, axis=1)
                mfccs_std = np.std(mfccs, axis=1)

                # Combine mean and std into a single array for each file
                if mfccs_mean.shape[0] == 13 and mfccs_std.shape[0] == 13:
                    combined_mfccs = np.stack([mfccs_mean, mfccs_std], axis=1)  # Shape: (13, 2)
                    features.append(combined_mfccs)
                    labels.append(df.at[f, 'Emotion'])
                    genders.append(df.at[f, 'Gender'])

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

        # Convert features and labels to arrays
        X = np.array(features)
        y = np.array(labels)
        genders = np.array(genders)

        # Flatten each (13, 2) array to (26,) so it can fit in a DataFrame
        X_flattened = np.array([x.flatten() for x in features])  # Shape: (num_samples, 26)
        
    
    def split_data_cnn(self, df):
        X = df.drop(columns=['Emotion', 'Gender']).values
        y = df['Emotion'].values

        # Reshape X for CNN input (samples, height, width, channels)
        # X = X.reshape(X.shape[0], 13, 2, 1)  # 13 MFCCs, 2 for mean and std, 1 for channel
        X = np.array(features).reshape(len(features), 13, 2, 1)

        # Encode labels
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
        
        return X, y, X_train, X_test, y_train, y_test
    
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
    
    




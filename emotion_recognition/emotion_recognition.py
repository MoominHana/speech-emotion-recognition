#imports
import os
import os
import librosa
import numpy as np
import pandas as pd

import tensorflow.keras as keras
import tensorflow as tf

#main
def main():

    model = tf.keras.models.load_model("speech_nn_model.keras")

    print("This is an emotion recognition system! (testing)")

    input_path = input("Enter file name: ")

    #if input is file
    if os.path.isfile(input_path):
        process_file(input_path)

    #if input is directory or folder
    elif os.path.isdir(input_path):
        process_folder(input_path)
    else:
        print("Invalid path.")

    data = process_data_nn(input_path)

    prediction = model.predict(data)

    print("Prediction:", prediction)

    
#functions
def process_file(file_path):
    
    print(f"Processing file: {file_path}")

def process_folder(folder_path):

    for filename in os.listdir(folder_path):
        if filename.endswith(".wav") or filename.endswith(".mp3"):
            file_path = os.path.join(folder_path, filename)
            process_file(file_path)

def process_data_nn(file):

        # Initialize lists for features and labels
        target_frames = 216
        # file_dir = "data/"

        # file = file_dir + file_name
        
        try:
            signal, rate = librosa.load(file, res_type='kaiser_fast',duration=2.5,sr=22050*2,offset=0.5) 
            # df.at[index, 'length'] = len(signal) / rate

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

        except Exception as e:
            print(f"Error processing file {file}: {e}")

        # Add batch dimension
        return np.expand_dims(mfccs, axis=0)

# for main - single comment
if __name__=="__main__":
    main()
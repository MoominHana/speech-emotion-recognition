#imports
import os
import librosa
import numpy as np
import pandas as pd
from pydub import AudioSegment
import sounddevice as sd
from scipy.io.wavfile import write
import time

import tensorflow.keras as keras
import tensorflow as tf

class System(object):
    
    #functions
    def process_file(self, file_path):

        # print(f"Processing file: {file_path}")
        print("")

    def process_folder(self, folder_path):

        for filename in os.listdir(folder_path):
            if filename.endswith(".wav") or filename.endswith(".mp3"):
                file_path = os.path.join(folder_path, filename)
                process_file(file_path)

    def process_data_nn(self, file):

            # Initialize lists for features and labels
            target_frames = 216
            # file_dir = "data/"

            # file = file_dir + file_name
            
            self.convert_to_wav(file)

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
                        
            # mfccs = np.expand_dims(mfccs, axis=-1) 

            except Exception as e:
                print(f"Error processing file {file}: {e}")

            # Add batch dimension
            return np.expand_dims(mfccs, axis=0)
        
    def convert_to_wav(self, input_file):
        # Check if the input file is already a wav file
        if input_file.lower().endswith('.wav'):
            # print(f"{input_file} is already a WAV file. No conversion needed.")
            return

        # Derive the output file name by replacing the input file extension with .wav
        base_name = os.path.splitext(input_file)[0]  # Get the file name without extension
        output_file = f"{base_name}.wav"

        try:
            # Load the audio file
            audio = AudioSegment.from_file(input_file)

            # Export the audio as a wav file
            audio.export(output_file, format="wav")
            print(f"Conversion complete! Saved as {output_file}")
        except Exception as e:
            print(f"An error occurred: {e}")
            
    def record_audio(self):
        duration=3
        fs=22050
        file_name = "recorded.wav"
        
        print(f"Recording for {duration} seconds...")
        audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Wait until the recording is finished
        print("Recording complete.")
        
        # Save the audio data as a .wav file
        write(file_name, fs, (audio_data * 32767).astype(np.int16))  # Scale to 16-bit PCM format
        # print(f"Audio saved as {file_name}")
    
        return file_name
    
    def main(self):
        
        class_map = {
            0: "angry",
            1: "calm",
            2: "disgust",
            3: "fearful",
            4: "happy",
            5: "neutral",
            6: "sad",
            7: "surprised"
        }


        model = tf.keras.models.load_model("speech_cnn_model.keras")

        print("This is an emotion recognition system!")

#         input_path = input("Enter file name: ")

#         #if input is file
#         if os.path.isfile(input_path):
#             self.process_file(input_path)

#         #if input is directory or folder
#         elif os.path.isdir(input_path):
#             self.process_folder(input_path)
#         else:
#             print("Invalid path.")

#         data = self.process_data_nn(input_path)
     

        while True:
            # Record a short audio snippet
            input_path = self.record_audio()
            
#             data = self.process_data_nn(audio_data)

            #if input is file
            if os.path.isfile(input_path):
                self.process_file(input_path)

            #if input is directory or folder
            elif os.path.isdir(input_path):
                self.process_folder(input_path)
            else:
                print("Invalid path.")

            data = self.process_data_nn(input_path)
        
            # print("Data shape: ", data.shape)

            prediction = model.predict(data)

            # print("Prediction:", prediction)
        
            # get index
            predicted_class = np.argmax(prediction, axis=1)[0]
            # print(f"Predicted Class Index: {predicted_class}")

            # get label
            predicted_label = class_map.get(predicted_class, "Unknown Class")
            print(f"Predicted Class: {predicted_label}")
            
            # Optionally, wait before recording again (this is like a "pause" before the next record)
            input("Press Enter to record again...")  # Wait for user input before recording next snippet

#             time.sleep(1) 

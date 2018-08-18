# animals = ['cat', 'dog', 'monkey']
# for animal in animals:
#     print(animal)

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
import numpy as np
import wave
import sys

import os

path = os.path.join('sources/audio1.wav')
print("path>>>" + path)
# spf = wave.open(path, 'rb')
#
# # Extract Raw Audio from Wav file
# signal = spf.readframes(-1)
# signal = np.frombuffer(signal, np.int16)
#
# # If Stereo
# # if spf.getnchannels() == 2:
# #     print('Just mono files')
# #     sys.exit(0)
#
# plt.figure(1)
# plt.title('Signal Wave...')
# plt.plot(signal)
# plt.show()
input_data = read(path)
audio = input_data[1]
# plot the first 1024 samples
plt.plot(audio[0:1024])
# label the axes
plt.ylabel("Amplitude")
plt.xlabel("Time")
# set the title
plt.title("Sample Wav")
# display the plot
plt.show()

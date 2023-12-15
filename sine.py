import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

num_points = 80  # Total number of points
sine_wave = np.sin(np.linspace(0, 4 * np.pi, num_points))

# Adjusting the sine wave to oscillate around 2.0 with an amplitude of 1.0
peak_value = 2.1
trough_value = 1.9
sine_wave = sine_wave * 0.1 + 2.0

df = pd.read_pickle('data/pre_train_df.pkl')
num_points = df.shape[0]
print(df.head())
sine_wave = df['bid_c']
# Plotting the sine wave as points
plt.figure(figsize=(8, 6))
plt.scatter(np.arange(num_points), sine_wave, s=5)  # Adjust s for point size
plt.title('Sine Wave Oscillating Around 2.0 (Points)')
plt.xlabel('Time')
plt.ylabel('Value')
plt.grid(True)
plt.show()
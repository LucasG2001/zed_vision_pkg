import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
folder_path = os.path.dirname(os.path.abspath(__file__))
# TODO: for parameter testing just do the "fake" timeseries here
if __name__ == '__main__':
    print("plotting")

    alpha = 0.1
    # Read data
    left_hand_df = pd.read_csv(os.path.join(folder_path, 'left_hand_data.csv'))
    right_hand_df = pd.read_csv(os.path.join(folder_path, 'right_hand_data.csv'))
    time_factor = 1/350

    # Function to calculate EMA
    def calculate_ema(data, alpha):
        return data.ewm(alpha=alpha, adjust=False).mean()

    # Calculate EMA for left hand data
    left_hand_df['x_ema'] = calculate_ema(left_hand_df['x'], alpha=alpha)
    left_hand_df['y_ema'] = calculate_ema(left_hand_df['y'], alpha=alpha)
    left_hand_df['z_ema'] = calculate_ema(left_hand_df['z'], alpha=alpha)

    # Calculate EMA for right hand data
    right_hand_df['x_ema'] = calculate_ema(right_hand_df['x'], alpha=alpha)
    right_hand_df['y_ema'] = calculate_ema(right_hand_df['y'], alpha=alpha)
    right_hand_df['z_ema'] = calculate_ema(right_hand_df['z'], alpha=alpha)

    # Plot original and EMA-filtered left hand data
    plt.figure(figsize=(18, 6))

    # Plot left hand data
    plt.subplot(2, 2, 1)
    plt.plot(left_hand_df.index*time_factor, left_hand_df['x'], label='x')
    plt.plot(left_hand_df.index*time_factor, left_hand_df['y'], label='y')
    plt.plot(left_hand_df.index*time_factor, left_hand_df['z'], label='z')
    plt.title('Left Hand Data vs Time')
    plt.xlabel('Time')
    plt.ylabel('Coordinate Value')
    plt.legend()

    # Plot EMA-filtered left hand data
    plt.subplot(2, 2, 2)
    plt.plot(left_hand_df.index*time_factor, left_hand_df['x_ema'], label='x EMA')
    plt.plot(left_hand_df.index*time_factor, left_hand_df['y_ema'], label='y EMA')
    plt.plot(left_hand_df.index*time_factor, left_hand_df['z_ema'], label='z EMA')
    plt.title('Left Hand EMA-filtered Data vs Time')
    plt.xlabel('Time')
    plt.ylabel('Coordinate Value')
    plt.legend()

    # Plot original and EMA-filtered right hand data
    plt.subplot(2, 2, 3)
    plt.plot(right_hand_df.index*time_factor, right_hand_df['x'], label='x')
    plt.plot(right_hand_df.index*time_factor, right_hand_df['y'], label='y')
    plt.plot(right_hand_df.index*time_factor, right_hand_df['z'], label='z')
    plt.title('Right Hand Data vs Time')
    plt.xlabel('Time')
    plt.ylabel('Coordinate Value')
    plt.legend()

    # Plot EMA-filtered right hand data
    plt.subplot(2, 2, 4)
    plt.plot(right_hand_df.index*time_factor, right_hand_df['x_ema'], label='x EMA')
    plt.plot(right_hand_df.index*time_factor, right_hand_df['y_ema'], label='y EMA')
    plt.plot(right_hand_df.index*time_factor, right_hand_df['z_ema'], label='z EMA')
    plt.title('Right Hand EMA-filtered Data vs Time')
    plt.xlabel('Time')
    plt.ylabel('Coordinate Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

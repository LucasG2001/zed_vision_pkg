import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


if __name__ == '__main__':
    print("plotting")

    folder_path = 'hand_data'
    left_hand_df = pd.read_csv(os.path.join(folder_path, 'left_hand_data.csv'), index_col=0)
    right_hand_df = pd.read_csv(os.path.join(folder_path, 'right_hand_data.csv'), index_col=0)

    # Plot left hand data
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(left_hand_df.index, left_hand_df['x'], label='x')
    plt.plot(left_hand_df.index, left_hand_df['y'], label='y')
    plt.plot(left_hand_df.index, left_hand_df['z'], label='z')
    plt.title('Left Hand Data vs Time')
    plt.xlabel('Time')
    plt.ylabel('Coordinate Value')
    plt.legend()

    # Plot right hand data
    plt.subplot(1, 2, 2)
    plt.plot(right_hand_df.index, right_hand_df['x'], label='x')
    plt.plot(right_hand_df.index, right_hand_df['y'], label='y')
    plt.plot(right_hand_df.index, right_hand_df['z'], label='z')
    plt.title('Right Hand Data vs Time')
    plt.xlabel('Time')
    plt.ylabel('Coordinate Value')
    plt.legend()

    plt.tight_layout()
    plt.show()

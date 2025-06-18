import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import os

def parse_tripinfo(xml_file):
    # Parses a SUMO tripinfo XML file and returns a pandas DataFrame
    if not os.path.exists(xml_file):
        print(f"Error: {xml_file} not found. Please run the evaluation scripts first.")
        return None

    tree = ET.parse(xml_file)
    root = tree.getroot()

    data = []
    for trip in root.findall('tripinfo'):
        trip_id = trip.get('id')
        duration = float(trip.get('duration'))
        waiting_time = float(trip.get('waitingTime'))
        data.append({'id': trip_id, 'duration': duration, 'waiting_time': waiting_time})

    return pd.DataFrame(data)

# --- Main Analysis ---
print("Analyzing evaluation results...")

# Parse both result files
df_fixed = parse_tripinfo('tripinfo_fixed.xml')
df_marl = parse_tripinfo('tripinfo_marl.xml')

if df_fixed is None or df_marl is None:
    print("Analysis cannot continue due to missing files.")
else:
    # Calculate key metrics
    metrics = {
        'Fixed-Time': {
            'Avg. Trip Duration (s)': df_fixed['duration'].mean(),
            'Avg. Wait Time (s)': df_fixed['waiting_time'].mean(),
            'Vehicles Completed': len(df_fixed)
        },
        'MARL': {
            'Avg. Trip Duration (s)': df_marl['duration'].mean(),
            'Avg. Wait Time (s)': df_marl['waiting_time'].mean(),
            'Vehicles Completed': len(df_marl)
        }
    }

    # Create a pandas DataFrame for easy printing and plotting
    metrics_df = pd.DataFrame(metrics).T

    # Print metrics to console
    print("\n--- Performance Comparison ---")
    print(metrics_df.round(2))

    # Create and save a bar chart
    plt.style.use('seaborn-v0_8-darkgrid')
    # Plot each metric in its own subplot
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(15, 5))
    fig.suptitle('Performance Comparison', fontsize=16)

    metrics_df['Avg. Trip Duration (s)'].plot(kind='bar', ax=axes[0], color=['skyblue', 'salmon'], rot=0)
    axes[0].set_ylabel('Seconds')
    axes[0].set_title('Average Trip Duration')

    metrics_df['Avg. Wait Time (s)'].plot(kind='bar', ax=axes[1], color=['skyblue', 'salmon'], rot=0)
    axes[1].set_ylabel('Seconds')
    axes[1].set_title('Average Wait Time')

    metrics_df['Vehicles Completed'].plot(kind='bar', ax=axes[2], color=['skyblue', 'salmon'], rot=0)
    axes[2].set_ylabel('Count')
    axes[2].set_title('Total Trips Completed')

    for ax in axes:
        ax.bar_label(ax.containers[0], fmt='%.1f')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust the layout
    plt.savefig('performance_comparison.png')
    print("\nComparison plot saved to 'performance_comparison.png'")
    plt.show()
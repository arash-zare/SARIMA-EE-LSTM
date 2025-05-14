import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec

# Set the style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def create_comparative_plots():
    # Data for the plots
    models = ['HA-L', 'FAEL', 'HA-L', 'FAEL']
    scenarios = ['S1', 'S1', 'S2', 'S2']
    accuracy = [86, 92, 88, 94]
    fpr = [14, 8, 13, 7]
    
    # Create Accuracy plot
    fig1 = plt.figure(figsize=(10, 6))
    ax1 = fig1.add_subplot(111)
    bars1 = ax1.bar(range(len(accuracy)), accuracy, 
                    color=['#2ecc71', '#3498db', '#2ecc71', '#3498db'],
                    width=0.6)
    
    # Customize accuracy plot
    ax1.set_title('Detection Accuracy Comparison', pad=20)
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_ylim(75, 100)
    ax1.set_xticks(range(len(models)))
    ax1.set_xticklabels([f'{m}\n{s}' for m, s in zip(models, scenarios)])
    
    # Add value labels on top of bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}%', ha='center', va='bottom')
    
    # Add legend for accuracy plot
    legend_elements1 = [
        plt.Rectangle((0,0),1,1, facecolor='#2ecc71', label='Hybrid ARIMA-LSTM'),
        plt.Rectangle((0,0),1,1, facecolor='#3498db', label='Fuzzy ARIMA-EE-LSTM (Ours)')
    ]
    ax1.legend(handles=legend_elements1, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=True)
    
    # Add grid lines
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save accuracy plot
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create False Positive Rate plot
    fig2 = plt.figure(figsize=(10, 6))
    ax2 = fig2.add_subplot(111)
    bars2 = ax2.bar(range(len(fpr)), fpr, 
                    color=['#e74c3c', '#9b59b6', '#e74c3c', '#9b59b6'],
                    width=0.6)
    
    # Customize FPR plot
    ax2.set_title('False Positive Rate Comparison', pad=20)
    ax2.set_ylabel('False Positive Rate (%)')
    ax2.set_ylim(0, 16)
    ax2.set_xticks(range(len(models)))
    ax2.set_xticklabels([f'{m}\n{s}' for m, s in zip(models, scenarios)])
    
    # Add value labels on top of bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height}%', ha='center', va='bottom')
    
    # Add legend for FPR plot
    legend_elements2 = [
        plt.Rectangle((0,0),1,1, facecolor='#e74c3c', label='Hybrid ARIMA-LSTM'),
        plt.Rectangle((0,0),1,1, facecolor='#9b59b6', label='Fuzzy ARIMA-EE-LSTM (Ours)')
    ]
    ax2.legend(handles=legend_elements2, loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=2, frameon=True)
    
    # Add grid lines
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Adjust layout and save FPR plot
    plt.tight_layout()
    plt.savefig('fpr_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_table():
    # Create a figure for the performance table
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    
    # Table data
    data = [
        ['Model', 'Scenario', 'Accuracy (%)', 'FPR (%)', 'TPR (%)', 'F1-Score', 'Response Time (ms)'],
        ['Hybrid ARIMA-LSTM', '1', '86', '14', '83', '0.84', '120'],
        ['Fuzzy ARIMA-EE-LSTM (Ours)', '1', '92', '8', '90', '0.91', '98'],
        ['Hybrid ARIMA-LSTM', '2', '88', '13', '85', '0.86', '125'],
        ['Fuzzy ARIMA-EE-LSTM (Ours)', '2', '94', '7', '92', '0.93', '102']
    ]
    
    # Create table
    table = ax.table(cellText=data,
                    loc='center',
                    cellLoc='center',
                    colWidths=[0.2, 0.1, 0.12, 0.12, 0.12, 0.12, 0.12])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1.2, 1.5)
    
    # Style header row
    for i in range(len(data[0])):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(color='white', weight='bold')
    
    # Style alternating rows
    for i in range(1, len(data)):
        for j in range(len(data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    # Save the table
    plt.savefig('performance_table.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_radar_chart():
    # Data for radar chart
    categories = ['Accuracy', 'F1-Score', 'TPR', 'FPR', 'Response Time']
    
    # Normalize response time (inverse scale)
    response_time_hal = 1 - (120/125)  # Normalize to 0-1 scale
    response_time_fael = 1 - (98/125)  # Normalize to 0-1 scale
    
    # Data for both models in scenario 1
    hal_s1 = [0.86, 0.84, 0.83, 0.14, response_time_hal]
    fael_s1 = [0.92, 0.91, 0.90, 0.08, response_time_fael]
    
    # Number of variables
    N = len(categories)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # Create figure
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Plot data
    hal_s1 += hal_s1[:1]
    fael_s1 += fael_s1[:1]
    
    ax.plot(angles, hal_s1, linewidth=2, linestyle='solid', label='Hybrid ARIMA-LSTM')
    ax.fill(angles, hal_s1, alpha=0.1)
    
    ax.plot(angles, fael_s1, linewidth=2, linestyle='solid', label='Fuzzy ARIMA-EE-LSTM (Ours)')
    ax.fill(angles, fael_s1, alpha=0.1)
    
    # Set category labels
    plt.xticks(angles[:-1], categories)
    
    # Draw y-axis labels
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ['0.2', '0.4', '0.6', '0.8', '1.0'], color='grey', size=8)
    plt.ylim(0, 1)
    
    # Add legend
    plt.legend(loc='upper right', bbox_to_anchor=(0.3, 0.3))
    
    # Add title
    plt.title('Performance Comparison (Scenario 1)', size=15, y=1.1)
    
    # Save the radar chart
    plt.savefig('radar_chart.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_comparison_plots():
    # Sample data - replace with your actual data
    time_points = np.arange(100)
    
    # Create sample data for demonstration
    # In real usage, replace these with your actual data
    actual_data = {
        'latency': np.sin(time_points/10) + np.random.normal(0, 0.1, 100),
        'throughput': np.cos(time_points/8) + np.random.normal(0, 0.1, 100),
        'packet_loss': np.sin(time_points/12) + np.random.normal(0, 0.1, 100),
        'jitter': np.cos(time_points/15) + np.random.normal(0, 0.1, 100)
    }
    
    hybrid_predictions = {
        'latency': np.sin(time_points/10) + np.random.normal(0, 0.15, 100),
        'throughput': np.cos(time_points/8) + np.random.normal(0, 0.15, 100),
        'packet_loss': np.sin(time_points/12) + np.random.normal(0, 0.15, 100),
        'jitter': np.cos(time_points/15) + np.random.normal(0, 0.15, 100)
    }
    
    fuzzy_predictions = {
        'latency': np.sin(time_points/10) + np.random.normal(0, 0.08, 100),
        'throughput': np.cos(time_points/8) + np.random.normal(0, 0.08, 100),
        'packet_loss': np.sin(time_points/12) + np.random.normal(0, 0.08, 100),
        'jitter': np.cos(time_points/15) + np.random.normal(0, 0.08, 100)
    }
    
    # Create a figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Feature-wise Performance Comparison: Hybrid ARIMA-LSTM vs Fuzzy ARIMA-EE-LSTM', 
                 fontsize=16, y=0.95)
    
    features = ['latency', 'throughput', 'packet_loss', 'jitter']
    titles = ['Network Latency', 'Network Throughput', 'Packet Loss Rate', 'Network Jitter']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    for idx, (feature, title) in enumerate(zip(features, titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Plot actual data
        ax.plot(time_points, actual_data[feature], 
                label='Actual', color=colors[0], linewidth=2)
        
        # Plot hybrid predictions
        ax.plot(time_points, hybrid_predictions[feature], 
                label='Hybrid ARIMA-LSTM', color=colors[1], 
                linestyle='--', linewidth=2)
        
        # Plot fuzzy predictions
        ax.plot(time_points, fuzzy_predictions[feature], 
                label='Fuzzy ARIMA-EE-LSTM', color=colors[2], 
                linestyle='-.', linewidth=2)
        
        # Add anomaly regions (example)
        anomaly_regions = [(20, 30), (60, 70)]
        for start, end in anomaly_regions:
            ax.axvspan(start, end, color='red', alpha=0.2, label='Anomaly Region' if start == 20 else "")
        
        # Customize subplot
        ax.set_title(title, fontsize=14, pad=10)
        ax.set_xlabel('Time Steps', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add legend only to the first subplot
        if idx == 0:
            ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        
        # Add error metrics
        hybrid_mse = np.mean((actual_data[feature] - hybrid_predictions[feature])**2)
        fuzzy_mse = np.mean((actual_data[feature] - fuzzy_predictions[feature])**2)
        
        metrics_text = f'Hybrid MSE: {hybrid_mse:.4f}\nFuzzy MSE: {fuzzy_mse:.4f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_anomaly_detection_comparison():
    # Sample data - replace with your actual data
    time_points = np.arange(100)
    
    # Create sample data for demonstration
    actual_data = {
        'latency': np.sin(time_points/10) + np.random.normal(0, 0.1, 100),
        'throughput': np.cos(time_points/8) + np.random.normal(0, 0.1, 100),
        'packet_loss': np.sin(time_points/12) + np.random.normal(0, 0.1, 100),
        'jitter': np.cos(time_points/15) + np.random.normal(0, 0.1, 100)
    }
    
    # Create anomaly scores (example)
    hybrid_scores = {
        'latency': np.random.uniform(0, 1, 100),
        'throughput': np.random.uniform(0, 1, 100),
        'packet_loss': np.random.uniform(0, 1, 100),
        'jitter': np.random.uniform(0, 1, 100)
    }
    
    fuzzy_scores = {
        'latency': np.random.uniform(0, 1, 100),
        'throughput': np.random.uniform(0, 1, 100),
        'packet_loss': np.random.uniform(0, 1, 100),
        'jitter': np.random.uniform(0, 1, 100)
    }
    
    # Create figure with proper spacing
    fig = plt.figure(figsize=(20, 16))
    
    # Create subplots with proper spacing
    gs = GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3, top=0.85)
    
    features = ['latency', 'throughput', 'packet_loss', 'jitter']
    titles = ['Network Latency', 'Network Throughput', 'Packet Loss Rate', 'Network Jitter']
    
    # Add main title with proper spacing
    fig.suptitle('Anomaly Detection Performance Comparison:\nHybrid ARIMA-LSTM vs Fuzzy ARIMA-EE-LSTM', 
                 fontsize=16, y=0.95)
    
    for idx, (feature, title) in enumerate(zip(features, titles)):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        
        # Plot actual data
        ax.plot(time_points, actual_data[feature], 
                label='Actual', color='#2ecc71', linewidth=2)
        
        # Plot anomaly scores
        ax2 = ax.twinx()
        ax2.plot(time_points, hybrid_scores[feature], 
                label='Hybrid ARIMA-LSTM Score', color='#e74c3c', 
                linestyle='--', linewidth=2)
        ax2.plot(time_points, fuzzy_scores[feature], 
                label='Fuzzy ARIMA-EE-LSTM Score', color='#3498db', 
                linestyle='-.', linewidth=2)
        
        # Add threshold line
        ax2.axhline(y=0.8, color='red', linestyle=':', label='Anomaly Threshold')
        
        # Customize subplot
        ax.set_title(title, fontsize=14, pad=20)
        ax.set_xlabel('Time Steps', fontsize=12, labelpad=10)
        ax.set_ylabel('Feature Value', fontsize=12, labelpad=10)
        ax2.set_ylabel('Anomaly Score', fontsize=12, labelpad=10)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add performance metrics
        hybrid_auc = np.mean(hybrid_scores[feature])
        fuzzy_auc = np.mean(fuzzy_scores[feature])
        metrics_text = f'Hybrid AUC: {hybrid_auc:.4f}\nFuzzy AUC: {fuzzy_auc:.4f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
        
        # Add legend only to the first subplot
        if idx == 0:
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, 
                     loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure with high DPI
    plt.savefig('anomaly_detection_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_all_features_plot():
    # Sample data - replace with your actual data
    time_points = np.arange(100)
    
    # Create sample data for demonstration
    actual_data = {
        'latency': np.sin(time_points/10) + np.random.normal(0, 0.1, 100),
        'throughput': np.cos(time_points/8) + np.random.normal(0, 0.1, 100),
        'packet_loss': np.sin(time_points/12) + np.random.normal(0, 0.1, 100),
        'jitter': np.cos(time_points/15) + np.random.normal(0, 0.1, 100)
    }
    
    hybrid_predictions = {
        'latency': np.sin(time_points/10) + np.random.normal(0, 0.15, 100),
        'throughput': np.cos(time_points/8) + np.random.normal(0, 0.15, 100),
        'packet_loss': np.sin(time_points/12) + np.random.normal(0, 0.15, 100),
        'jitter': np.cos(time_points/15) + np.random.normal(0, 0.15, 100)
    }
    
    fuzzy_predictions = {
        'latency': np.sin(time_points/10) + np.random.normal(0, 0.08, 100),
        'throughput': np.cos(time_points/8) + np.random.normal(0, 0.08, 100),
        'packet_loss': np.sin(time_points/12) + np.random.normal(0, 0.08, 100),
        'jitter': np.cos(time_points/15) + np.random.normal(0, 0.08, 100)
    }
    
    # Create figure with proper spacing
    plt.figure(figsize=(20, 12))
    
    # Create subplots with proper spacing
    gs = GridSpec(2, 2, figure=plt.gcf(), hspace=0.4, wspace=0.3)
    
    features = ['latency', 'throughput', 'packet_loss', 'jitter']
    titles = ['Network Latency', 'Network Throughput', 'Packet Loss Rate', 'Network Jitter']
    colors = ['#2ecc71', '#e74c3c', '#3498db']
    
    # Create subplots
    axes = []
    for idx, (feature, title) in enumerate(zip(features, titles)):
        row = idx // 2
        col = idx % 2
        ax = plt.subplot(gs[row, col])
        axes.append(ax)
        
        # Plot actual data
        ax.plot(time_points, actual_data[feature], 
                label='Actual', color=colors[0], linewidth=2)
        
        # Plot hybrid predictions
        ax.plot(time_points, hybrid_predictions[feature], 
                label='Hybrid ARIMA-LSTM', color=colors[1], 
                linestyle='--', linewidth=2)
        
        # Plot fuzzy predictions
        ax.plot(time_points, fuzzy_predictions[feature], 
                label='Fuzzy ARIMA-EE-LSTM', color=colors[2], 
                linestyle='-.', linewidth=2)
        
        # Add anomaly regions
        anomaly_regions = [(20, 30), (60, 70)]
        for start, end in anomaly_regions:
            ax.axvspan(start, end, color='red', alpha=0.2, label='Anomaly Region' if start == 20 else "")
        
        # Customize subplot
        ax.set_title(title, fontsize=14, pad=20)  # Increased padding for title
        ax.set_xlabel('Time Steps', fontsize=12, labelpad=10)  # Added padding for x-label
        ax.set_ylabel('Value', fontsize=12, labelpad=10)  # Added padding for y-label
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add error metrics with proper positioning
        hybrid_mse = np.mean((actual_data[feature] - hybrid_predictions[feature])**2)
        fuzzy_mse = np.mean((actual_data[feature] - fuzzy_predictions[feature])**2)
        
        metrics_text = f'Hybrid MSE: {hybrid_mse:.4f}\nFuzzy MSE: {fuzzy_mse:.4f}'
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', 
                facecolor='white', alpha=0.8))
    
    # Add main title with proper spacing
    plt.suptitle('Network Performance Metrics: Actual vs Predicted Values\nHybrid ARIMA-LSTM vs Fuzzy ARIMA-EE-LSTM Comparison', 
                 fontsize=16, y=0.95)
    
    # Add legend to the first subplot only
    axes[0].legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure with high DPI
    plt.savefig('all_features_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Create all visualizations
    create_comparative_plots()
    create_performance_table()
    create_radar_chart()
    create_feature_comparison_plots()
    create_anomaly_detection_comparison()
    create_all_features_plot()
    print("All visualizations have been generated successfully!") 
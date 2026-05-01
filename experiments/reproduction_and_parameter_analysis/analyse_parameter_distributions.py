import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('experiment_results.csv')

# Group the phases so we don't run out of shape markers
def group_phase(p):
    p = str(p)
    if p.startswith('1'): return 'Random'
    if p.startswith('2'): return 'Boundary'
    if p.startswith('3'): return 'Topology'
    return 'Other'

df['Phase_Group'] = df['Phase'].apply(group_phase)

# Add random vertical jitter to simulate stacking and reveal density
np.random.seed(42)
df['dummy_y'] = np.random.uniform(-0.5, 0.5, size=len(df))

# The 12 variables we swept
params = ['c_ee', 'c_ei', 'c_ie', 'c_ii', 'P', 'Q', 'tau_1', 'tau_2', 'rho', 'beta', 'k', 'alpha']

fig, axes = plt.subplots(4, 3, figsize=(20, 16))
axes = axes.flatten()

# Custom colors (Green for True, Red for False)
palette = {True: 'green', False: 'red'}

print(f"Plotting {len(df)} data points...")

for i, param in enumerate(params):
    if param in df.columns:
        sns.scatterplot(
            data=df, 
            x=param, 
            y='dummy_y', 
            hue='Oscillation_Detected', 
            style='Phase_Group', 
            palette=palette, 
            alpha=0.6, 
            ax=axes[i],
            edgecolor='k', 
            linewidth=0.2
        )
        axes[i].set_title(f'{param} Distribution')
        axes[i].set_yticks([]) # Hide the meaningless Y-axis
        axes[i].set_ylabel('')
        
        # Clean up the legends (only show it on the first plot to save space)
        if i > 0 and axes[i].legend_:
            axes[i].get_legend().remove()
        elif i == 0:
            axes[i].legend(loc='upper right', bbox_to_anchor=(1.4, 1.0))

plt.tight_layout()
plt.show()
# plt.savefig('parameter_exploration_latest.png', dpi=150)
# print("Saved latest snapshot to parameter_exploration_latest.png!")
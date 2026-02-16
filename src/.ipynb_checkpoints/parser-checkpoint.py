import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_csv(path):
    """Load DatLab CSV file."""
    df = pd.read_csv(path)
    return df

def extract_flux(df, chamber, plot_name="O2 flux"):
    """Extract flux values for a specific chamber and plot."""
    return df[(df["chamber"] == chamber) & (df["plot name"] == plot_name)]

def get_state_flux(df):
    """Get key respiration states (LEAK, OXPHOS, ETS) for both chambers."""
    states = {
        "LEAK": "Add oligomycin",
        "OXPHOS": "Add ADP",
        "ETS": "Add FCCP"
    }

    summary = {"State": [], "Chamber 1": [], "Chamber 2": []}
    
    for state, event in states.items():
        summary["State"].append(state)
        for chamber in ["Chamber 1", "Chamber 2"]:
            value = df[(df["chamber"]==chamber) & 
                       (df["plot name"]=="O2 flux") & 
                       (df["event name"]==event)]["value"].values[0]
            summary[chamber].append(value)
    return summary

def compute_ratios(summary):
    """Compute OXPHOS/LEAK and ETS/OXPHOS ratios."""
    efficiency = {"OXPHOS/LEAK": [], "ETS/OXPHOS": []}
    
    for chamber in ["Chamber 1", "Chamber 2"]:
        LEAK, OXPHOS, ETS = summary[chamber]
        efficiency["OXPHOS/LEAK"].append(OXPHOS / LEAK)
        efficiency["ETS/OXPHOS"].append(ETS / OXPHOS)
    return efficiency

def plot_flux(df):
    """Plot O2 flux over time for both chambers with event labels."""
    ch1 = extract_flux(df, "Chamber 1")
    ch2 = extract_flux(df, "Chamber 2")
    
    plt.figure(figsize=(10,6))
    plt.plot(ch1["time [min]"], ch1["value"], marker='o', label='Chamber 1', color='blue')
    plt.plot(ch2["time [min]"], ch2["value"], marker='s', label='Chamber 2', color='red')
    
    for _, row in df[df["plot name"]=="O2 flux"].drop_duplicates(subset=["time [min]", "event name"]).iterrows():
        plt.axvline(x=row["time [min]"], color='gray', linestyle='--', alpha=0.5)
        plt.text(row["time [min]"]+0.1, max(ch1["value"])+2, row["event name"], rotation=45, fontsize=9)
    
    plt.xlabel("Time (min)")
    plt.ylabel("O2 flux (pmol·s⁻¹·mg⁻¹)")
    plt.title("O2 Flux Over Time - Both Chambers")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_bar(summary):
    """Plot key respiration states as bar graph."""
    states_labels = summary["State"]
    ch1_values = np.array(summary["Chamber 1"])
    ch2_values = np.array(summary["Chamber 2"])
    
    x = np.arange(len(states_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x - width/2, ch1_values, width, label='Chamber 1', color='blue')
    rects2 = ax.bar(x + width/2, ch2_values, width, label='Chamber 2', color='red')

    ax.set_ylabel('O2 flux (pmol·s⁻¹·mg⁻¹)')
    ax.set_xlabel('Respiration State')
    ax.set_title('Key Respiration States by Chamber')
    ax.set_xticks(x)
    ax.set_xticklabels(states_labels)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.tight_layout()
    plt.show()

def plot_ratios(efficiency):
    """Plot respiration efficiency ratios as bar graph."""
    ratios_labels = list(efficiency.keys())
    ch1_ratios = [efficiency[key][0] for key in ratios_labels]
    ch2_ratios = [efficiency[key][1] for key in ratios_labels]

    x = np.arange(len(ratios_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x - width/2, ch1_ratios, width, label='Chamber 1', color='blue')
    rects2 = ax.bar(x + width/2, ch2_ratios, width, label='Chamber 2', color='red')

    ax.set_ylabel('Ratio')
    ax.set_xlabel('Respiration Efficiency')
    ax.set_title('Mitochondrial Respiration Ratios')
    ax.set_xticks(x)
    ax.set_xticklabels(ratios_labels)
    ax.legend()

    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom')
    plt.tight_layout()
    plt.show()
def summarize_replicates(df):
    """
    Summarize respiration states across replicates.
    Returns a dict with mean and std for each state and chamber.
    """
    states = {
        "LEAK": "Add oligomycin",
        "OXPHOS": "Add ADP",
        "ETS": "Add FCCP"
    }
    
    chambers = df["chamber"].unique()
    
    summary_stats = {ch: {"State": [], "Mean": [], "Std": []} for ch in chambers}
    
    for state, event in states.items():
        for ch in chambers:
            # Get all flux values for this event/chamber
            values = df[(df["chamber"]==ch) & 
                        (df["plot name"]=="O2 flux") & 
                        (df["event name"]==event)]["value"].values
            summary_stats[ch]["State"].append(state)
            summary_stats[ch]["Mean"].append(values.mean())
            summary_stats[ch]["Std"].append(values.std())
    
    return summary_stats
    
def plot_bar_with_error_save(summary_stats, filename):
    #Plot respiration states with mean ± SD and save to file.
    import matplotlib.pyplot as plt
    import numpy as np

    chambers = list(summary_stats.keys())
    states_labels = summary_stats[chambers[0]]["State"]
    
    x = np.arange(len(states_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8,5))
    
    for i, ch in enumerate(chambers):
        means = summary_stats[ch]["Mean"]
        stds = summary_stats[ch]["Std"]
        ax.bar(x + (i - len(chambers)/2)*width, means, width, yerr=stds, label=ch, capsize=5)
    
    ax.set_ylabel("O2 flux (pmol·s⁻¹·mg⁻¹)")
    ax.set_xlabel("Respiration State")
    ax.set_title("Respiration States Across Replicates")
    ax.set_xticks(x)
    ax.set_xticklabels(states_labels)
    ax.legend()
    plt.tight_layout()
    
    # Save figure to file
    fig.savefig(filename)
    plt.close(fig)

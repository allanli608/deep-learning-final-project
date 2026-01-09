import json
import matplotlib.pyplot as plt

# ==========================================
# CONFIGURATION
# ==========================================
# Change this variable to point to your JSON file path
INPUT_FILE = "report_data/trainer_state good model.json"
# ==========================================


def plot_training_logs(file_path):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{file_path}'.")
        return

    # Extract log history
    # The JSON structure you provided typically has the list inside 'log_history'
    history = data.get("log_history", [])

    if not history:
        print("No 'log_history' found in the file.")
        return

    # Initialize lists to store data
    steps = []
    losses = []
    lrs = []
    grad_norms = []

    # Iterate through history and collect data
    # We check if keys exist because sometimes logs contain eval metrics (eval_loss)
    # but not training metrics for a specific step.
    for entry in history:
        if "loss" in entry and "step" in entry:
            steps.append(entry["step"])
            losses.append(entry["loss"])

            # These are usually paired with loss, but we use .get() just in case
            lrs.append(entry.get("learning_rate", 0))
            grad_norms.append(entry.get("grad_norm", 0))

    if not steps:
        print("No training steps with 'loss' data found.")
        return

    # Create the plots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # 1. Loss Plot
    ax1.plot(steps, losses, label="Training Loss", color="tab:blue", linewidth=2)
    ax1.set_ylabel("Loss")
    ax1.set_title("Training Loss over Steps")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.legend()

    # 2. Learning Rate Plot
    ax2.plot(steps, lrs, label="Learning Rate", color="tab:orange", linewidth=2)
    ax2.set_ylabel("Learning Rate")
    ax2.set_title("Learning Rate Schedule")
    ax2.grid(True, linestyle="--", alpha=0.7)
    ax2.legend()

    # 3. Gradient Norm Plot
    ax3.plot(steps, grad_norms, label="Gradient Norm", color="tab:green", linewidth=1.5)
    ax3.set_xlabel("Global Step")
    ax3.set_ylabel("Grad Norm")
    ax3.set_title("Gradient Norms")
    ax3.grid(True, linestyle="--", alpha=0.7)
    ax3.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # For testing purposes, you can create a dummy file if one doesn't exist
    # logic to create a dummy file for demonstration:
    import os

    if not os.path.exists(INPUT_FILE):
        print(f"'{INPUT_FILE}' not found. Creating a sample file for demonstration...")
        sample_data = {
            "log_history": [
                {
                    "step": 100,
                    "loss": 1.603,
                    "learning_rate": 1.98e-5,
                    "grad_norm": 2.84,
                },
                {
                    "step": 200,
                    "loss": 0.4745,
                    "learning_rate": 1.97e-5,
                    "grad_norm": 1.57,
                },
                {
                    "step": 300,
                    "loss": 0.3500,
                    "learning_rate": 1.96e-5,
                    "grad_norm": 1.20,
                },
                {
                    "step": 400,
                    "loss": 0.2800,
                    "learning_rate": 1.95e-5,
                    "grad_norm": 0.95,
                },
            ]
        }
        with open(INPUT_FILE, "w") as f:
            json.dump(sample_data, f)

    plot_training_logs(INPUT_FILE)

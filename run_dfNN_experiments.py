import torch
import matplotlib.pyplot as plt

# Import in the order we like
from simulate import (
    simulate_convergence,
    simulate_branching,
    simulate_merge,
    simulate_deflection,
    simulate_ridge,
)

# Define simulations
simulations = {
    "convergence": simulate_convergence,
    "branching": simulate_branching,
    "merge": simulate_merge,
    "deflection": simulate_deflection,
    "ridge": simulate_ridge,
}

# Load training inputs
x_train = torch.load("data/sim_data/x_train_lines_discretised.pt").float()

# Storage dictionaries
y_train_dict = {}
test_predictions_dict = {}

# Iterate over all simulation functions
for name, sim_func in simulations.items():
    # Generate training observations
    y_train = sim_func(x_train)
    y_train_dict[name] = y_train  # Store training outputs

    # Print details
    print(f"=== {name.upper()} ===")
    print(f"Training inputs shape: {x_train.shape}")
    print(f"Training observations shape: {y_train.shape}")
    print(f"Training inputs dtype: {x_train.dtype}")
    print()

    # Generate test predictions (example using x_train as test input)
    test_predictions = sim_func(x_train)
    test_predictions_dict[name] = test_predictions

# Save a training plot for one case (e.g., "convergence")
plt.figure(figsize=(6, 4))
plt.plot(x_train.numpy(), y_train_dict["convergence"].numpy(), "o", label="Training Data")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Training Data for Convergence")
plt.legend()
plt.savefig("training_plot_convergence.png")
plt.show()

# Save test predictions
torch.save(test_predictions_dict, "test_predictions.pt")

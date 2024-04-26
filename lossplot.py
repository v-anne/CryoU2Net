import matplotlib.pyplot as plt
import re

def extract_loss(file_path):
    # This pattern matches only the loss values from the relevant lines.
    pattern = re.compile(r"Train Loss: ([\d.]+), Validation Loss: ([\d.]+),")

    train_losses = []
    validation_losses = []

    try:
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    train_loss = float(match.group(1))
                    validation_loss = float(match.group(2))
                    train_losses.append(train_loss)
                    validation_losses.append(validation_loss)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
        return [], []
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return [], []

    if not train_losses:
        print("No data was extracted. Check the file content and regex pattern.")

    return train_losses, validation_losses

def plot_losses(train_losses, validation_losses):
    if not train_losses:
        print("No data available to plot.")
        return
    
    epochs = range(1, len(train_losses) + 1)  # Assuming epochs are sequential and start at 1
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.plot(epochs, validation_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

# Path to your .out file
file_path = 'slurm-2178808.out'

# Extract data
train_losses, validation_losses = extract_loss(file_path)

# Plot the data
plot_losses(train_losses, validation_losses)

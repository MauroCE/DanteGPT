import pickle
import matplotlib.pyplot as plt
from matplotlib import rc


# with open("losses/model1_training.pkl", "rb") as file:
#     training_losses = pickle.load(file)
#
# with open("losses/model1_validation.pkl", "rb") as file:
#     validation_losses = pickle.load(file)
with open("losses/model2_big_training_500_30000.pkl", "rb") as file:
    training_losses = pickle.load(file)

with open("losses/model2_big_validation_500_30000.pkl", "rb") as file:
    validation_losses = pickle.load(file)


# Train and Validation
rc('font', **{'family': 'STIXGeneral'})
fig, ax = plt.subplots()
ax.plot(training_losses[::500], label='Train', color='indianred', marker='x')
ax.plot(validation_losses, label='Validation', color='lightseagreen', marker='+')
ax.set_xlabel("Iteration")
ax.set_ylabel("Cross-Entropy Loss")
ax.grid(True, color='gainsboro')
ax.set_title("Model 2 (Small): Train and Validation Losses every 500th iteration")
ax.legend()
# plt.savefig("images/model2_smaller_train_val_losses.png")
plt.show()

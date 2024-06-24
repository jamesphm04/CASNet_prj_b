import matplotlib.pyplot as plt
import numpy as np

# Define the confusion matrix data
TP = 116
TN = 86
FP = 2
FN = 1

# Create the confusion matrix
conf_matrix = np.array([[TP, FN], [FP, TN]])
plt.figure(figsize=(6, 6))
# Plot the confusion matrix
plt.imshow(conf_matrix, cmap='Blues', interpolation='nearest')
#plt.title('Confusion Matrix')
# Add labels to the plot
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['Non-Deformed', 'Deformed'], fontsize=16)
plt.yticks(tick_marks, ['Non-Deformed', 'Deformed'], rotation=90, fontsize=16, verticalalignment='center')

# Add text annotations
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(conf_matrix[i, j]), horizontalalignment='center', verticalalignment='center', fontsize=32)

plt.ylabel('Actual label', fontsize=22)
plt.xlabel('Predicted label', fontsize=22)
#plt.colorbar()
Recall = TP/(TP+FN)
precision = TP/(TP+FP)
F1= 2*(precision*Recall)/(precision+Recall)
print("F1: "+str(F1))
print("Recall: "+str(Recall))
print("Precision: "+str(precision))

plt.show()

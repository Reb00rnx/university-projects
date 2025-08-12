import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

# Switch to non-GUI backend
matplotlib.use('Agg')

# Load the confusion matrix CSV file
file_path = '/home/patryk/awaryjny/trip/results/confusion_matrix.csv'
confusion_matrix = pd.read_csv(file_path, index_col=0)

# Select only the first 17 rows and columns
confusion_matrix_17 = confusion_matrix.iloc[:17, :17]

# Create a heatmap for the filtered confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_matrix_17, annot=True, cmap='Blues', fmt='.2f')

plt.title('Confusion Matrix for 17 Sections')
plt.xlabel('Predicted label')
plt.ylabel('True label')

# Save the heatmap
plt.savefig('/home/patryk/awaryjny/trip/results/confusion_matrix_17_sections.png')


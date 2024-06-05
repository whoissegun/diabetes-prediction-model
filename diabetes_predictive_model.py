import torch
class DiabetesPredictionModelV1(torch.nn.Module): #binary classification model
    def __init__(self, in_features, out_features):
      super().__init__()
      self.layers = torch.nn.Sequential(
          torch.nn.Linear(in_features = in_features, out_features= 16 ),
          torch.nn.BatchNorm1d(16),
          torch.nn.ReLU(),
          torch.nn.Dropout(p=0.25),
          torch.nn.Linear(in_features = 16, out_features = 32 ),
          torch.nn.BatchNorm1d(32),
          torch.nn.ReLU(),
          torch.nn.Dropout(p=0.25),
          torch.nn.Linear(in_features = 32, out_features = 16 ),
          torch.nn.BatchNorm1d(16),
          torch.nn.ReLU(),
          torch.nn.Dropout(p=0.25),
          torch.nn.Linear(in_features = 16, out_features = 8 ),
          torch.nn.BatchNorm1d(8),
          torch.nn.ReLU(),
          torch.nn.Dropout(p=0.25),
          torch.nn.Linear(in_features = 8, out_features = 4 ),
          torch.nn.BatchNorm1d(4),
          torch.nn.ReLU(),
          torch.nn.Dropout(p=0.25),
          torch.nn.Linear(in_features = 4, out_features = out_features ),
      )

    def forward(self, X):
      return self.layers(X).squeeze(dim=1)
    
if __name__ == "__main__":

  import opendatasets as od
  import pandas as pd
  import torch
  from sklearn.preprocessing import StandardScaler

  DATASET_URL="https://www.kaggle.com/datasets/ehababoelnaga/diabetes-dataset/data"

  od.download(DATASET_URL)

  train_df = pd.read_csv("/content/diabetes-dataset/Training.csv")
  test_df = pd.read_csv("/content/diabetes-dataset/Testing.csv")

  # Drop the Outcome column to get features
  train_features = train_df.drop(["Outcome"], axis=1)
  test_features = test_df.drop(["Outcome"], axis=1)

  # Store the Outcome column separately as target
  train_target = train_df["Outcome"]
  test_target = test_df["Outcome"]

  #drop rows with missing data
  train_features.dropna()
  test_features.dropna()

  #standardize the columns
  scaler = StandardScaler()
  train_features_scaled = scaler.fit_transform(train_features) #this standardizes each column but returns a numpy array

  test_features_scaled = scaler.fit_transform(test_features)

  #convert the data to tensors
  train_features_tensors = torch.Tensor(train_features_scaled)
  test_features_tensors = torch.Tensor(test_features_scaled)

  train_target_tensors = torch.Tensor(train_target)
  test_target_tensors = torch.Tensor(test_target)

  

  model = DiabetesPredictionModelV1(train_features_tensors.shape[1], 1)

  loss_fn = torch.nn.BCEWithLogitsLoss()
  optim = torch.optim.Adam(params=model.parameters(), lr=0.03, weight_decay=0.02)

  from sklearn.metrics import confusion_matrix, accuracy_score

  epochs = 500
  best_accuracy = 0
  best_confusion_matrix = None
  for epoch in range(epochs):
    model.train()
    train_preds = model(train_features_tensors)
    train_loss = loss_fn(train_preds, train_target_tensors)
    optim.zero_grad()
    train_loss.backward()
    optim.step() #this updates the models parameters

    if epoch % 10 == 0:
      model.eval()
      with torch.no_grad():  # No need to track gradients during evaluation
              test_preds = model(test_features_tensors).squeeze()
              test_preds_class = (torch.sigmoid(test_preds) > 0.5).float()  # Convert logits to binary class predictions

              # Calculate accuracy
              accuracy = accuracy_score(test_target_tensors.numpy(), test_preds_class.numpy())
              if(accuracy > best_accuracy):
                best_accuracy = accuracy
                best_confusion_matrix = confusion_matrix(test_target_tensors.numpy(), test_preds_class.numpy())


  print(f'Best Test Accuracy: {best_accuracy:.4f}')
  print("Best confusion matrix:")
  print(best_confusion_matrix)

  import matplotlib.pyplot as plt
  import seaborn as sns

  # To visualize the confusion matrix with labels
  fig, ax = plt.subplots()
  sns.heatmap(best_confusion_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
  ax.set_xlabel('Predicted Labels')
  ax.set_ylabel('True Labels')
  ax.set_title('Confusion Matrix')
  ax.xaxis.set_ticklabels(['Negative (0)', 'Positive (1)'])
  ax.yaxis.set_ticklabels(['Negative (0)', 'Positive (1)'])
  plt.show()

  torch.save(model,"model.pth")
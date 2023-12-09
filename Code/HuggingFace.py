import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import numpy as np

# Load your dataset
df = pd.read_csv('final_data.csv')

# Modify df
df = df[['lyrics', 'danceability', 'energy', 'key', 'mode', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', 'track_popularity']]
df.drop_duplicates(subset=['lyrics'], keep='first', ignore_index=True)
df['target'] = df['track_popularity'].apply(lambda x: 0 if x <= 49 else 1)
df = df.drop(columns=['track_popularity'])

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenize text data
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
train_encodings = tokenizer(train_df['lyrics'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(test_df['lyrics'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Standardize additional features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_df.iloc[:, 1:-1])
test_features = scaler.transform(test_df.iloc[:, 1:-1])

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, encodings, additional_features, labels):
        self.encodings = encodings
        self.additional_features = additional_features
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}  # Corrected
        item['additional_features'] = torch.tensor(self.additional_features[idx], dtype=torch.float32)
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)

# Create Train and Test CustomDataset instances
train_dataset = CustomDataset(train_encodings, train_features, train_df['target'].values)
test_dataset = CustomDataset(test_encodings, test_features, test_df['target'].values)

# Define Data Loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

        
# Custom Model
class CustomBertModel(nn.Module):
    def __init__(self, num_features):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.classifier = nn.Linear(self.bert.config.hidden_size + num_features, 2)

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined_features = torch.cat((pooled_output, features), dim=1)
        logits = self.classifier(combined_features)
        return logits

model = CustomBertModel(num_features=train_features.shape[1])

# Training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)


# Training Loop
for epoch in range(3):  # Number of epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        additional_features = batch['additional_features'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids, attention_mask, additional_features)
        loss = nn.CrossEntropyLoss()(outputs, labels)  # Corrected
        loss.backward()
        optimizer.step()

# Evaluation Function
def evaluate_model(model, data_loader, device):
    model.eval()
    total, correct, all_predictions, all_labels = 0, 0, [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, additional_features)
            predictions = torch.argmax(outputs, dim=1)

            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return accuracy, f1




# Evaluation
accuracy, f1 = evaluate_model(model, test_loader, device)
print(f"Model accuracy on the test set: {accuracy:.2f}")
print(f"F1 Score on the test set: {f1:.2f}")


# Save the model
torch.save(model.state_dict(), 'model.pth')
print("Model saved successfully.")

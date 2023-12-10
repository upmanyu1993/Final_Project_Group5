import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Load dataset
try:
    df = pd.read_csv('final_data.csv')
except FileNotFoundError:
    print("File not found. Please check the file path.")
    exit()

# Drop non-numeric columns
cols_to_drop = ['language', 'Total_syllables', 'syllable_per_line', 'syllable_per_word', 'syllable_var', 'novel_word_proportion', 
                'rhymes_per_line', 'rhymes_per_syllable', 'end_pairs_per_line', 'end_pairs_variation', 'average_end_score', 'average_end_syl_score']
df.drop(cols_to_drop, axis=1, inplace=True)

# Rename 'lyrics1' to 'lyrics' if necessary
df.rename(columns={'lyrics1': 'lyrics'}, inplace=True)

# Split dataset into train and test sets
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Identify numeric features for scaling
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Apply StandardScaler only to numeric features
scaler = StandardScaler()
train_df[numeric_features] = scaler.fit_transform(train_df[numeric_features])
test_df[numeric_features] = scaler.transform(test_df[numeric_features])

# Extract numeric columns for PCA
numeric_features = [col for col in train_df.columns if col.isdigit()]
pca = PCA(n_components=300)
train_embeddings = pca.fit_transform(train_df[numeric_features])
test_embeddings = pca.transform(test_df[numeric_features])

# Create DataFrames from the embeddings and concatenate with original DataFrames
train_embeddings_df = pd.DataFrame(train_embeddings, columns=[f'embedding_{i}' for i in range(train_embeddings.shape[1])])
test_embeddings_df = pd.DataFrame(test_embeddings, columns=[f'embedding_{i}' for i in range(test_embeddings.shape[1])])
train_df = pd.concat([train_df, train_embeddings_df], axis=1)
test_df = pd.concat([test_df, test_embeddings_df], axis=1)

# Drop original numeric features
train_df.drop(numeric_features, axis=1, inplace=True)
test_df.drop(numeric_features, axis=1, inplace=True)

# Tokenize text data
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
train_encodings = tokenizer(train_df['lyrics'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")
test_encodings = tokenizer(test_df['lyrics'].tolist(), truncation=True, padding=True, max_length=512, return_tensors="pt")

# Identify numeric features excluding 'lyrics' and target column for scaling
numeric_features = train_df.select_dtypes(include=[np.number]).columns.tolist()

# Standardize only numeric features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_df[numeric_features])
test_features = scaler.transform(test_df[numeric_features])

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, encodings, additional_features, labels):
        self.encodings = encodings
        self.additional_features = additional_features
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
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

class CustomBertModel(nn.Module):
    def __init__(self, num_features):
        super(CustomBertModel, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size + num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 4)  
        )

    def forward(self, input_ids, attention_mask, features):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        combined_features = torch.cat((pooled_output, features), dim=1)
        combined_features = self.dropout(combined_features)
        logits = self.classifier(combined_features)
        # Softmax to get probabilities for 4 classes
        softmax_logits = nn.functional.softmax(logits, dim=1)
        return softmax_logits

# Initialize and train the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CustomBertModel(num_features=train_features.shape[1]).to(device)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)

def evaluate_model(model, data_loader, device):
    model.eval()
    total_loss, total, correct, all_labels, all_predictions = 0, 0, 0, [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            additional_features = batch['additional_features'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask, additional_features)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    average_loss = total_loss / len(data_loader)
    accuracy = correct / total
    f1 = f1_score(all_labels, all_predictions, average='weighted')
    return average_loss, accuracy, f1

# Training Loop
train_losses = []
val_losses = []
for epoch in range(5):
    model.train()
    train_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        additional_features = batch['additional_features'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask, additional_features)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader)
    train_losses.append(train_loss)

    val_loss, val_accuracy = evaluate_model(model, test_loader, device)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Plot Training and Validation Losses
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()
plt.show()

# Final Evaluation
accuracy, f1 = evaluate_model(model, test_loader, device)
print(f"Model accuracy on the test set: {accuracy:.2f}")
print(f"F1 Score on the test set: {f1:.2f}")

# Save the Model
torch.save(model.state_dict(), 'model.pth')
print("Model saved successfully.")

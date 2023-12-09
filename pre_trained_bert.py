import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from transformers import AutoConfig, AutoTokenizer
from transformers import BertPreTrainedModel, BertModel
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('final_data.csv')
df.drop_duplicates(['track_name', 'track_artist'], inplace=True)

df.drop_duplicates(['lyrics.1'], inplace=True)
scaler = StandardScaler()
popularity_2d = df['track_popularity'].values.reshape(-1, 1)

# Apply StandardScaler
scaler = StandardScaler()
df['track_popularity'] = scaler.fit_transform(popularity_2d)

# Split the data
train_data, temp_data = train_test_split(df, test_size=0.2, random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
from torch import nn

class BertRegresser(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        #The output layer that takes the [CLS] representation and gives an output
        self.cls_layer1 = nn.Linear(config.hidden_size,128)
        self.relu1 = nn.ReLU()
        self.ff1 = nn.Linear(128,128)
        self.tanh1 = nn.Tanh()
        self.ff2 = nn.Linear(128,1)

    def forward(self, input_ids, attention_mask):
        #Feed the input to Bert model to obtain contextualized representations
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        #Obtain the representations of [CLS] heads
        logits = outputs.last_hidden_state[:,0,:]
        output = self.cls_layer1(logits)
        output = self.relu1(output)
        output = self.ff1(output)
        output = self.tanh1(output)
        output = self.ff2(output)
        return output


def prepare_data_loader(df, tokenizer, batch_size):
    lyrics = df['lyrics.1'].tolist()
    targets = (df['track_popularity']).to_numpy()
        
    inputs = tokenizer(lyrics, padding=True, truncation=True, return_tensors="pt", max_length=512)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], torch.tensor(targets))
    
    return DataLoader(dataset, batch_size=batch_size)

## Configuration loaded from AutoConfig 
config = AutoConfig.from_pretrained('bert-base-uncased')

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# Load the pre-trained BERT model
model = BertRegresser.from_pretrained('bert-base-uncased', config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.to(device)

# Create an instance of the custom model
# Check if GPU is available and set the device accordingly
print(f"Using device: {device}")
# model = CustomBERTModel(bert_model)
# model.to(device)

batch_size = 16
train_loader = prepare_data_loader(train_data, tokenizer, batch_size)
val_loader = prepare_data_loader(val_data, tokenizer, batch_size)
test_loader = prepare_data_loader(test_data, tokenizer, batch_size)

from torch.optim import Adam
from torch.nn import MSELoss
from torch.optim.lr_scheduler import StepLR
from torch.optim import Adam

# Define optimizer and scheduler
# Define loss function and optimizer
loss_function = MSELoss()

optimizer = Adam(model.parameters(), lr=5e-5)
scheduler = StepLR(optimizer, step_size=1, gamma=0.1)  # Adjust as needed

def train_model(model, train_loader, val_loader, device, epochs):
    from tqdm import tqdm
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        train_loss = 0

        # Training loop
        for batch in train_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            optimizer.zero_grad()

            logits = model(input_ids, attention_mask=attention_mask).squeeze(-1)
            loss = loss_function(logits, labels.float())
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]

                logits = model(input_ids, attention_mask=attention_mask).squeeze(-1)
                loss = loss_function(logits, labels.float())
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}")

# Example training call
train_model(model, train_loader, val_loader, device, epochs=5)

def evaluate_model(model, data_loader, device):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]

            # Directly use the output tensor
            outputs = model(input_ids, attention_mask=attention_mask)
            batch_predictions = outputs.squeeze(-1).cpu().numpy()  # Squeeze if the output has an extra dimension
            batch_labels = labels.cpu().numpy()

            predictions.extend(batch_predictions)
            actuals.extend(batch_labels)

    mse = mean_squared_error(actuals, predictions)
    mae = mean_absolute_error(actuals, predictions)
    r_squared = r2_score(actuals, predictions)
    explained_variance = explained_variance_score(actuals, predictions)

    return mse, mae, r_squared, explained_variance

# Call evaluate_model with your test_loader
mse, mae, r_squared, explained_variance = evaluate_model(model, test_loader, device)
print(f"Test MSE: {mse}, Test MAE: {mae}, Test R²: {r_squared}, Test Explained Variance: {explained_variance}")

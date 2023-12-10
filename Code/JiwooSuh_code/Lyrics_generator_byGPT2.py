from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import Trainer, TrainingArguments
import pandas as pd
import torch
#%%
data = pd.read_csv('spotify_songs 2.csv')
# data = pd.read_csv('final_data.csv')
print(data.info())
print(data.head())
df = data[['track_id', 'track_name', 'track_artist', 'lyrics', 'language', 'track_popularity']]
df = df[df['language'] == 'en']
df.drop(['language'], axis=1, inplace=True)
df.dropna(subset=['lyrics'], inplace=True)
print(df.info())

#%%
# Queen_df = df[df['track_artist'] == 'Queen']['lyrics']
# Queen_df.reset_index()
# print(Queen_df.head())
# Queen_df.to_csv('queendf.csv', header = None, index=False)

#%%
artist_lyrics_count = df.groupby('track_artist')['lyrics'].count().reset_index()

artist_lyrics_count.columns = ['track_artist', 'lyrics_count']

artist_lyrics_count = artist_lyrics_count.sort_values(by='lyrics_count', ascending=False)
artist_lyrics_count10 = artist_lyrics_count[:10]
artist_lists = artist_lyrics_count10.track_artist.to_list()

print(artist_lyrics_count10)
print(artist_lists)

#%%
def create_artist_lyrics_csv(artist_name, input_csv, output_csv):
    data = pd.read_csv(input_csv)

    print(data.info())
    print(data.head())

    df = data[['track_id', 'track_name', 'track_artist', 'lyrics', 'language', 'track_popularity']]
    df = df[df['language'] == 'en']
    df.drop(['language'], axis=1, inplace=True)
    df.dropna(subset=['lyrics'], inplace=True)

    artist_df = df[df['track_artist'] == artist_name]['lyrics']
    artist_df.reset_index(drop=True, inplace=True)
    print(artist_df.head())

    artist_df.to_csv(output_csv, header=None, index=False)

for artist in artist_lists:
    output_csv = f"{artist.lower().replace(' ', '_')}_df.csv"
    create_artist_lyrics_csv(artist, 'spotify_songs 2.csv', output_csv)

#%%
def load_dataset(file_path, tokenizer, block_size=128):
    dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=block_size,
    )
    return dataset


def load_data_collator(tokenizer, mlm=False):
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=mlm,
    )
    return data_collator


def train(train_file_path, model_name,
          output_dir,
          overwrite_output_dir,
          per_device_train_batch_size,
          num_train_epochs,
          save_steps):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    train_dataset = load_dataset(train_file_path, tokenizer)
    data_collator = load_data_collator(tokenizer)

    tokenizer.save_pretrained(output_dir)

    model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

    model.save_pretrained(output_dir)

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=overwrite_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        num_train_epochs=num_train_epochs,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    trainer.save_model()

#%%
def train_all_models():
    for artist in artist_lists:
        input_csv = f"{artist.lower().replace(' ', '_')}_df.csv"
        output_model_name = f"gpt2_{artist.lower().replace(' ', '_')}"
        output_dir = f"results_{artist.lower().replace(' ', '_')}"
        train_file_path = input_csv
        overwrite_output_dir = False
        per_device_train_batch_size = 8
        num_train_epochs = 5.0
        save_steps = 500

        train(train_file_path, 'gpt2', output_dir, overwrite_output_dir,
                    per_device_train_batch_size, num_train_epochs, save_steps)

# train_all_models() # UNCOMMENT NEEDED

#%%
# train_file_path = "queendf.csv"
# model_name = 'gpt2'
# output_dir = 'results'
# overwrite_output_dir = False
# per_device_train_batch_size = 8
# num_train_epochs = 5.0
# save_steps = 500

# train(
#     train_file_path=train_file_path,
#     model_name=model_name,
#     output_dir=output_dir,
#     overwrite_output_dir=overwrite_output_dir,
#     per_device_train_batch_size=per_device_train_batch_size,
#     num_train_epochs=num_train_epochs,
#     save_steps=save_steps
# )

#%%
from transformers import PreTrainedTokenizerFast, GPT2LMHeadModel, GPT2TokenizerFast, GPT2Tokenizer
def load_model(model_path):
    model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
    return model


def load_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    return tokenizer


def generate_text(model_path, sequence, max_length):
    model_path = model_path
    model = load_model(model_path)
    tokenizer = load_tokenizer(model_path)
    ids = tokenizer.encode(f'{sequence}', return_tensors='pt').to(device)
    final_outputs = model.generate(
        ids,
        do_sample=True,
        max_length=max_length,
        pad_token_id=model.config.eos_token_id,
        top_k=50,
        top_p=0.95,
    )
    return(tokenizer.decode(final_outputs[0], skip_special_tokens=True))

#%%
import os
model_base_path = 'results_'
result_list = []
device = "cuda" if torch.cuda.is_available() else "cpu"

for artist in artist_lists:
    model_path = model_base_path + f"{artist.lower().replace(' ', '_')}"
    sequence = 'love is'
    max_len = 300
    print(f"{artist}")
    generated_lyrics = generate_text(model_path, sequence, max_len)

    result_dict = {
        'artist': artist,
        'sequence': sequence,
        'generated_lyrics': generated_lyrics
    }

    result_list.append(result_dict)

    print(result_dict)

# print(result_list)


# sequence = 'love is'
# max_len = 300
# generate_text(sequence, max_len)

#%%
# Model Evaluation - Human Judgement


#%%
# Model Evaluation - Lyric Similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def read_csv_to_string(filename):
    with open(filename) as f:
        text = f.readlines()
        text = ' '.join(text)
    return text

for result_dict in result_list:
    artist = result_dict['artist']
    generated_lyrics = result_dict['generated_lyrics']
    sequence = result_dict['sequence']
    real_lyrics = read_csv_to_string(f"{artist.lower().replace(' ', '_')}_df.csv")
    lyrics = [generated_lyrics, real_lyrics]
    encoded_input = tokenizer(lyrics, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
    formatted_similarity = '{:.3f}'.format(similarity.item())
    print(f"Cosine similarity between real and generated lyrics of {artist}: {formatted_similarity}")


def run_lyrics_generator(model_base_path, artist_lists, sequence):
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
            num_beams=5,
            pad_token_id=model.config.eos_token_id,
            early_stopping=True,
            no_repeat_ngram_size=5,
            top_k=50,
            top_p=0.95,
        )
        return(tokenizer.decode(final_outputs[0], skip_special_tokens=True))
    
    import os
    import torch
    # model_base_path = 'results_'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    result_list = []
    # artist_lists = ['david_guetta']
    for artist in artist_lists:
        model_path = model_base_path + f"{artist.lower().replace(' ', '_')}"
        # sequence = 'love is'
        max_len = 500
        print(f"{artist}")
        generated_lyrics = generate_text(model_path, sequence, max_len)
    
        result_dict = {
            'artist': artist,
            'sequence': sequence,
            'generated_lyrics': generated_lyrics
        }
    
        result_list.append(result_dict)
    
    return result_dict['generated_lyrics']


def read_csv_to_string(filename):
    with open(filename) as f:
        text = f.readlines()
        text = ' '.join(text)
    return text

def calculate_cosine_similarity(artist, generated_lyrics):
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer, AutoModel
    import torch.nn.functional as F
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2').to(device)
    real_lyrics = read_csv_to_string(f"train_{artist.lower().replace(' ', '_')}_dataset.csv")

    lyrics = [generated_lyrics, real_lyrics]
    encoded_input = tokenizer(lyrics, padding=True, truncation=True, return_tensors='pt').to(device)
    with torch.no_grad():
        model_output = model(**encoded_input)

    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    similarity = F.cosine_similarity(sentence_embeddings[0].unsqueeze(0), sentence_embeddings[1].unsqueeze(0))
    formatted_similarity = '{:.3f}'.format(similarity.item())
    return formatted_similarity

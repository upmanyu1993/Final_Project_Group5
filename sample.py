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
            pad_token_id=model.config.eos_token_id,
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
        max_len = 1000
        print(f"{artist}")
        generated_lyrics = generate_text(model_path, sequence, max_len)
    
        result_dict = {
            'artist': artist,
            'sequence': sequence,
            'generated_lyrics': generated_lyrics
        }
    
        result_list.append(result_dict)
    
    return result_dict['generated_lyrics']
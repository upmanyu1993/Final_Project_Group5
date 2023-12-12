
from keybert import KeyBERT
# Function to get summarized lyrics
def get_summarized_lyrics(lyrics, summarizer):
    summary = summarizer(lyrics, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

def get_summarization(lyrics, tokenizer, model):
    inputs = tokenizer(lyrics, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=30, num_beams=4, length_penalty=2.0, early_stopping=True)
    summarization = tokenizer.batch_decode(summary_ids, skip_special_tokens=True)
    return summarization


def get_keywords(lyrics, keyphrase_ngram_range=(1, 1), stop_words='english'):
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(lyrics, keyphrase_ngram_range=keyphrase_ngram_range, stop_words=stop_words)
    return keywords



# # Apply summarization to the DataFrame
# df['summarized_lyrics'] = df['lyrics'].apply(get_summarized_lyrics)
#
# # Display the DataFrame with the new 'sentiment' and 'summarized_lyrics' columns
# print(df.head())
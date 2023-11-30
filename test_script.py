import pandas as pd
df = pd.read_csv('spotify_songs.csv')

from deep_translator import GoogleTranslator
import textwrap
def google_translate(text):
    """Translates a song if it is not in english using google translator"""
    google_translator = GoogleTranslator()
    # If text length is less than 5k chars we directly translate
    # If not we divide in 5000 chars slots and translate them separately, then join
    if len(text) <= 5000:
        lyrics = google_translator.translate(text)
        return lyrics
    else:
        segments = textwrap.wrap(text, 5000)
        translation = []
        for segment in segments:
            translation.append(google_translator.translate(segment))
        return ' '.join(translation)

df.dropna(subset = 'lyrics',inplace=True)
df.index = range(len(df))

non_eng_df = df[df['language']!='en']
non_eng_df.index = range(len(non_eng_df))
from tqdm import tqdm
for i in tqdm(range(len(non_eng_df))):
    non_eng_df.loc[i,'new_lyrics'] = google_translate(non_eng_df.loc[i,'lyrics'][0:4999])

    




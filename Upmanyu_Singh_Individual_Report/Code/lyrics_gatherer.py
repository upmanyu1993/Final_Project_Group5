from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
import langdetect
import textwrap
import random
import lyricsgenius

GENIUS_TOKEN = 'mbyvV9TfHHLOffKsiuI9-uRH4j8cjtIBzN3F9YxzWY-72tHQLGYtFAI_Pqg-yJ2b'

class LyricsGatherer:

    def __init__(self):
        self.model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        self.title = None
        self.artist = None
        self.lyrics = None
        self.embedding = None

    def retrieve_lyrics(self, title, artist):
        """Gets raw data of a song given its title and artist"""
        genius = lyricsgenius.Genius(access_token=GENIUS_TOKEN)
        self.title = title
        self.artist = artist
        try:
            lyrics = genius.search_song(title, artist).lyrics
            self.lyrics = lyrics
            return lyrics
        except Exception as e:
            print(e)
            return None

    def google_translate(self, text):
        """Translates a song if it is not in english using google translator"""
        try:
            if langdetect.detect(text) == 'en':
                self.lyrics = text
                return text
        except:
            self.lyrics = text
            return text

        google_translator = GoogleTranslator()
        # If text length is less than 5k chars we directly translate
        # If not we divide in 5000 chars slots and translate them separately, then join
        if len(text) <= 5000:
            lyrics = google_translator.translate(text)
            self.lyrics = lyrics
            return lyrics
        else:
            segments = textwrap.wrap(text, 5000)
            translation = []
            for segment in segments:
                translation.append(google_translator.translate(segment))
            self.lyrics = ' '.join(translation)
            return ' '.join(translation)

    def encode(self, lyrics):
        """Transform lyrics into vectors"""
        embedding = self.model.encode(lyrics, normalize_embeddings=True)
        self.embedding = embedding
        return embedding

    def get_formatted_lyrics(self, title, artist):
        """Gets a vector of song lyrics of a song"""
        text = self.retrieve_lyrics(title, artist)
        if text != None:
            text = self.google_translate(text)
            return self.encode(text)
        else:
            return None




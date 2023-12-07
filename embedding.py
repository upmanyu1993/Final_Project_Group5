# import pandas as pd
# import multiprocessing
# from tqdm import tqdm
# import lyrics_gatherer

# def gather_lyrics(track_info):
#     try:
#         print(f"Processing {track_info}...")  # Logging
#         lg = lyrics_gatherer.LyricsGatherer()
#         track_name, track_artist = track_info
#         formatted_lyrics = lg.get_formatted_lyrics(track_name, track_artist)
#         raw_lyrics = lg.lyrics
#         print(f"Done with {track_info}.")  # Logging
#         return formatted_lyrics, raw_lyrics
#     except Exception as e:
#         print(f"Error processing {track_info}: {e}")
#         return None, None

# def main():
#     df = pd.read_csv('spotify_songs2.csv')
#     track_info_list = [(row['track_name'], row['track_artist']) for index, row in df.iterrows()]

#     with multiprocessing.Pool(processes=4) as pool:  # Reduced number of processes
#         results = list(tqdm(pool.imap(gather_lyrics, track_info_list), total=len(track_info_list)))

#     embeddings, lyrics = zip(*results)
#     # Further processing...

# if __name__ == '__main__':
#     main()


# import pandas as pd
# import concurrent.futures
# from tqdm import tqdm

# def test_function(track_info):
#     print(f"Processing {track_info}...")
#     return "Done"

# def main():
#     df = pd.read_csv('spotify_songs2.csv')
#     track_info_list = [(row['track_name'], row['track_artist']) for index, row in df.iterrows()]

#     with concurrent.futures.ProcessPoolExecutor() as executor:
#         results = list(tqdm(executor.map(test_function, track_info_list), total=len(track_info_list)))

#     print(results)

# if __name__ == '__main__':
#     main()
import lyrics_gatherer

lg = lyrics_gatherer.LyricsGatherer()

# lg stores the lyrics in its own variables
# raw lyrics
import pandas as pd
import numpy as np
df = pd.read_csv('spotify_songs3.csv')
tem = df[df['lyrics.1']=='Not Found']
tem.index= range(len(tem))
# df.loc[797,'lyrics']
# temp = lg.get_formatted_lyrics('let me enter',)
# df = df.iloc[0:10,:]
from tqdm import tqdm
embedding = []
lyrics = []
for i in tqdm(range(len(tem))):
    # i = 797
    try:
        temp = lg.get_formatted_lyrics(tem.loc[i,'track_name'], tem.loc[i,'track_artist'])
        if  temp is None:
            embedding.append(np.array([np.nan]))
            lyrics.append('Not Found')
        else:
            embedding.append(temp)
            lyrics.append(lg.lyrics.split('Lyrics')[-1])
    except:
        embedding.append(np.array([np.nan]))
        lyrics.append('Not Found')

final = pd.DataFrame(embedding)
# final['embedding'] = embedding
final['lyrics'] = lyrics

df = pd.concat([df, final],axis=1)
df.to_csv('spotify_songs3.csv',index=False)
# encoded lyrics
# (can you use multiprocessing in this )


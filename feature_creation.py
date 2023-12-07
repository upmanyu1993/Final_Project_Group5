import pandas as pd
import syllapy
from nltk.tokenize import sent_tokenize
import numpy as np
import pronouncing
df = pd.read_csv('final_data.csv')
df.columns
df.drop_duplicates(subset = ['track_name','track_artist'],inplace=True)
df.drop_duplicates(subset = ['lyrics.1'],inplace=True)
lyrics = pd.read_csv('lyrics_feature.csv')
lyrics.drop_duplicates(subset=['track_name','track_artist'],inplace=True)
# lyrics.drop_duplicates(subset = ['lyrics.1'],inplace=True)
lyrics.index=range(len(lyrics))
lyrics = pd.merge(df[['track_name','track_artist','track_popularity']], lyrics, on=['track_name','track_artist'], how='right')
lyrics.to_csv('lyrics_feature.csv',index=False)

audio = pd.read_csv('audio_feature.csv')
audio.drop_duplicates(subset=['track_name','track_artist'],inplace=True)
audio = pd.merge(df[['track_name','track_artist','track_popularity']], audio, on=['track_name','track_artist'], how='right')
audio.to_csv('audio_feature.csv',index=False)

sentence = pd.read_csv('sentence_embedding.csv')
sentence.drop_duplicates(subset=['track_name','track_artist'],inplace=True)
sentence  = pd.merge(df[['track_name','track_artist','track_popularity']], sentence, on=['track_name','track_artist'], how='right')
sentence.to_csv('sentence_embedding.csv',index=False)

# df.to_csv('final_data.csv',index=False)
# audio_feature = pd.concat([df[df.columns[1:3]],df[df.columns[12:24]], df['track_popularity']],axis=1)
# audio_feature.to_csv('audio_feature.csv',index=False)
# lyrics_feature = pd.concat([df[df.columns[1:3]],df[df.columns[36:804]], df['track_popularity']],axis=1)
# lyrics_feature.to_csv('sentence_embedding.csv',index=False)

# df.index=range(len(df))
# tem = df[df['lyrics.1']=='Not Found']


def syllables_per_line(lyrics):
    lines = lyrics.split('\n')
    return [sum(syllapy.count(w) for w in line.split()) for line in lines]
    
def sent_tok(lyrics):
    sent = sent_tokenize(lyrics)
    return '\n'.join(sent)

def syllables_per_word(lyrics):
    words = lyrics.split()
    syllable_counts = [syllapy.count(word) for word in words]
    return sum(syllable_counts) / len(syllable_counts) if words else 0

def syllable_variation(lyrics):
    syllable_counts = syllables_per_line(lyrics)
    return np.std(syllable_counts) if syllable_counts else 0

def novel_word_proportion(lyrics):
    lines = lyrics.split('\n')
    novel_word_props = []
    for i in range(len(lines) - 1):
        words_first = set(lines[i].split())
        words_second = set(lines[i + 1].split())
        novel_words = words_second - words_first
        if words_second:
            novel_word_props.append(len(novel_words) / len(words_second))
    return np.mean(novel_word_props)

def rhymes_per_line(lyrics):
    lines = lyrics.split('\n')
    rhyme_counts = []
    for line in lines:
        words = line.split()
        rhymes = [w for w in words if pronouncing.rhymes(w)]
        rhyme_counts.append(len(rhymes))
    return rhyme_counts

def rhymes_per_syllable(lyrics):
    lines = lyrics.split('\n')
    total_syllables = 0
    total_rhymes = 0

    for line in lines:
        words = line.split()
        syllables = sum(syllapy.count(w) for w in words)
        total_syllables += syllables

        rhymes = [w for w in words if pronouncing.rhymes(w)]
        total_rhymes += len(rhymes)

    return total_rhymes / total_syllables if total_syllables > 0 else 0

from collections import defaultdict

def rhyme_density(lyrics):
    lines = lyrics.split('\n')
    total_syllables = 0
    rhymed_syllables = 0
    rhyme_cache = defaultdict(list)
    from tqdm import tqdm
    for line in tqdm(lines):
        words = line.split()
        syllable_counts = [syllapy.count(w) for w in words]
        total_syllables += sum(syllable_counts)

        for i, word in enumerate(words):
            if word not in rhyme_cache:
                rhyme_cache[word] = pronouncing.rhymes(word)

            rhymes = rhyme_cache[word]
            if any(w in rhymes for w in words[:i] + words[i+1:]):
                rhymed_syllables += syllable_counts[i]

    return rhymed_syllables / total_syllables if total_syllables > 0 else 0

def clean_word(word):
    # Remove punctuation and convert to lower case
    return ''.join(char for char in word if char.isalpha()).lower()

def end_pairs_per_line(lyrics):
    lines = lyrics.split('\n')
    total_lines = len(lines) - 1  # Adjust for the loop range
    end_rhyme_count = 0

    for i in range(total_lines):
        line_current = lines[i].split()
        line_next = lines[i + 1].split()

        if line_current and line_next:  # Check if both lines are not empty
            last_word_current = clean_word(line_current[-1])
            last_word_next = clean_word(line_next[-1])

            # Check for rhymes
            if last_word_next and last_word_current and last_word_next in pronouncing.rhymes(last_word_current):
                end_rhyme_count += 1

    return (end_rhyme_count / total_lines) * 100 if total_lines > 0 else 0

def end_pairs_variation(lyrics):
    lines = lyrics.split('\n')
    total_couplets = len(lines) - 1
    grown, shrunk, even = 0, 0, 0

    for i in range(total_couplets):
        words_current = lines[i].split()
        words_next = lines[i + 1].split()

        if words_current and words_next:  # Check if both lines have words
            syllables_current = syllapy.count(words_current[-1])
            syllables_next = syllapy.count(words_next[-1])

            if syllables_next > syllables_current * 1.15:  # 15% longer
                grown += 1
            elif syllables_next < syllables_current * 0.85:  # 15% shorter
                shrunk += 1
            else:
                even += 1
        else:
            # If one of the lines is empty, consider it as 'even' for the sake of calculation
            even += 1

    grown_percent = (grown / total_couplets) * 100 if total_couplets > 0 else 0
    shrunk_percent = (shrunk / total_couplets) * 100 if total_couplets > 0 else 0
    even_percent = (even / total_couplets) * 100 if total_couplets > 0 else 0

    return grown_percent, shrunk_percent, even_percent
def phonetic_similarity(word1, word2):
    phones1 = pronouncing.phones_for_word(word1)
    phones2 = pronouncing.phones_for_word(word2)
    if not phones1 or not phones2:
        return 0

    phones1 = phones1[0].split()
    phones2 = phones2[0].split()

    # Compare from the end to the start
    similarity = 0
    for p1, p2 in zip(reversed(phones1), reversed(phones2)):
        if p1 == p2:
            similarity += 1
        else:
            break

    return similarity

def average_end_score(lyrics):
    lines = lyrics.split('\n')
    total_score = 0
    count = 0

    for i in range(len(lines) - 1):
        words_current = lines[i].split()
        words_next = lines[i + 1].split()

        if words_current and words_next:  # Ensure both lines have words
            last_word_current = words_current[-1]
            last_word_next = words_next[-1]

            if last_word_next in pronouncing.rhymes(last_word_current):
                total_score += phonetic_similarity(last_word_current, last_word_next)
                count += 1

    return total_score / count if count > 0 else 0

def average_end_syl_score(lyrics):
    lines = lyrics.split('\n')
    total_score = 0
    total_syllables = 0

    for i in range(len(lines) - 1):
        words_current = lines[i].split()
        words_next = lines[i + 1].split()

        # Ensure both lines have words before proceeding
        if words_current and words_next:
            last_word_current = words_current[-1]
            last_word_next = words_next[-1]

            if last_word_next in pronouncing.rhymes(last_word_current):
                score = phonetic_similarity(last_word_current, last_word_next)
                syllables = syllapy.count(last_word_current) + syllapy.count(last_word_next)
                total_score += score
                total_syllables += syllables

    return total_score / total_syllables if total_syllables > 0 else 0

# temp = df.head(100)
def count_rhyme_lengths(lyrics):
    lines = lyrics.split('\n')
    rhyme_lengths = {1: 0, 2: 0, 3: 0, 4: 0, 'longs': 0}
    
    # Caching the rhymes and syllable counts
    rhymes_cache = {}
    syllable_count_cache = {}

    for line in lines:
        words = line.split()
        for word in words:
            if word not in rhymes_cache:
                rhymes_cache[word] = pronouncing.rhymes(word)

            for rhyme in rhymes_cache[word]:
                if rhyme not in syllable_count_cache:
                    syllable_count_cache[rhyme] = syllapy.count(rhyme)

                syllable_count = syllable_count_cache[rhyme]
                if syllable_count == 1:
                    rhyme_lengths[1] += 1
                elif syllable_count == 2:
                    rhyme_lengths[2] += 1
                elif syllable_count == 3:
                    rhyme_lengths[3] += 1
                elif syllable_count == 4:
                    rhyme_lengths[4] += 1
                else:
                    rhyme_lengths['longs'] += 1

    total_rhymes = sum(rhyme_lengths.values())
    if total_rhymes == 0:
        return {length: 0 for length in rhyme_lengths}

    # Convert counts to percentages
    rhyme_lengths_percent = {length: (count / total_rhymes) * 100 for length, count in rhyme_lengths.items()}
    return rhyme_lengths_percent


# df['lyrics'] = df['lyrics'].apply(lambda x: sent_tok(x))
df['syllable_per_line'] = df['lyrics.1'].apply(lambda x: np.mean(syllables_per_line(x)))
df['syllable_per_word'] = df['lyrics.1'].apply(lambda x: syllables_per_word(x))
df['syllable_var'] = df['lyrics.1'].apply(lambda x: syllable_variation(x))
df['novel_word_proportion'] = df['lyrics.1'].apply(lambda x: novel_word_proportion(x))
df['rhymes_per_line'] = df['lyrics.1'].apply(lambda x: np.mean(rhymes_per_line(x)))
df['rhymes_per_syllable'] = df['lyrics.1'].apply(lambda x: rhymes_per_syllable(x))
df['rhyme_density'] = df['lyrics.1'].apply(lambda x: rhyme_density(x))
df['end_pairs_per_line'] = df['lyrics.1'].apply(lambda x: end_pairs_per_line(x))
df['end_pairs_variation'] = df['lyrics.1'].apply(lambda x: end_pairs_variation(x))
df['average_end_score'] = df['lyrics.1'].apply(lambda x: average_end_score(x))
df['average_end_syl_score'] = df['lyrics.1'].apply(lambda x: average_end_syl_score(x))
df['count_rhyme_lengths'] = df['lyrics.1'].apply(lambda x: count_rhyme_lengths(x))

import ast

lyrics['count_rhyme_lengths'] = lyrics['count_rhyme_lengths'].apply(lambda x: ast.literal_eval(x))
for i in range(len(lyrics)):
    dic = lyrics.loc[i,'count_rhyme_lengths']    
    lyrics.at[i,'count_rhyme_lengths_1'] = dic[1]
    lyrics.at[i,'count_rhyme_lengths_2'] = dic[2]
    lyrics.at[i,'count_rhyme_lengths_3'] = dic[3]
    lyrics.at[i,'count_rhyme_lengths_4'] = dic[4]
    lyrics.at[i,'count_rhyme_lengths_longs'] = dic['longs']

lyrics.drop(['count_rhyme_lengths'],axis=1,inplace=True)

def expand_list_to_columns(data, max_length):
    # Ensure the data is a list and pad it if necessary
    data = list(data) if data else []
    data += [None] * (max_length - len(data))
    return data[:max_length]

lyrics['end_pairs_variation'] = lyrics['end_pairs_variation'].apply(lambda x: x.replace('(','').replace(')','').split(','))


# Apply the function to each column and create new columns
# for col in ['syllable_per_line', 'novel_word_proportion', 'end_pairs_variation','rhymes_per_line']:
col = 'end_pairs_variation'
expanded_cols = lyrics[col].apply(lambda x: expand_list_to_columns(x, 3))
expanded_col_names = [f'{col}_{i+1}' for i in range(3)]
lyrics[expanded_col_names] = pd.DataFrame(expanded_cols.tolist(), index=lyrics.index)
lyrics[expanded_col_names] = lyrics[expanded_col_names].astype(float)
lyrics.drop(['end_pairs_variation'],axis=1,inplace=True)
lyrics.to_csv('lyrics_feature.csv',index=False)

# import prosodic as p
# import concurrent.futures

# # Configure prosodic for non-verbose mode
# p.config['print_to_screen'] = 0

# def classify_meter(stress_pattern):
#     meters = {
#         'iambic': '01',
#         'trochaic': '10',
#         'spondaic': '11',
#         'anapestic': '001',
#         'dactylic': '100',
#         'amphibrachic': '010',
#         'pyrrhic': '00'
#     }

#     for meter, pattern in meters.items():
#         if pattern in stress_pattern:
#             return meter
#     return 'unknown'

# def process_line(line, stress_str2int):
#     stress_pattern = ''.join([str(stress_str2int[s]) for s in line.str_stress()])
#     return classify_meter(stress_pattern)

# def meter_percentages(text, timeout=5):  # Set the timeout in seconds
#     pText = p.Text(text)
#     pText.parse()
#     meter_counts = {meter: 0 for meter in ['iambic', 'trochaic', 'spondaic', 'anapestic', 'dactylic', 'amphibrachic', 'pyrrhic', 'unknown']}

#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         futures = []
#         for line in pText.lines():
#             futures.append(executor.submit(process_line, line, line.stress_str2int))

#         for future in concurrent.futures.as_completed(futures, timeout=timeout):
#             try:
#                 meter = future.result()
#                 meter_counts[meter] += 1
#             except concurrent.futures.TimeoutError:
#                 print("Processing line timed out")
#                 meter_counts['unknown'] += 1
#             except Exception as e:
#                 print(f"Error processing line: {e}")
#                 meter_counts['unknown'] += 1

#     total_lines = sum(meter_counts.values())
#     meter_percentages = {meter: (count / total_lines) * 100 for meter, count in meter_counts.items()} if total_lines > 0 else meter_counts
    
#     return meter_percentages
# # print (df.head())
# df.index=range(len(df))

# from tqdm import tqdm
# for i in tqdm(range(len(df))):
#     # print (i)
#     # i=55
#     meter = meter_percentages(df.loc[i,'lyrics.1'])
#     df.at[i,'meter_analysis_iambic'] = meter['iambic']
#     df.at[i,'meter_analysis_trochaic'] = meter['trochaic']
#     df.at[i,'meter_analysis_spondaic'] = meter['spondaic']
#     df.at[i,'meter_analysis_anapestic'] = meter['anapestic']
#     df.at[i,'meter_analysis_dactylic'] = meter['dactylic']
#     df.at[i,'meter_analysis_amphibrachic'] = meter['amphibrachic']
#     df.at[i,'meter_analysis_pyrrhic'] = meter['pyrrhic']
#     df.at[i,'meter_analysis_unknown'] = meter['unknown']

# df.to_csv('/content/drive/MyDrive/spotify_songs3.csv',index=False)


# df11 = pd.DataFrame(np.concatenate([df.iloc[:,12:24], df.iloc[:,26:]],axis=1), columns = np.concatenate([df.iloc[:,12:24].columns, df.iloc[:,26:].columns]))
# df11['track_popularity'] = df['track_popularity'].tolist()
# df = df11.copy()


# # Drop the original columns if they are no longer needed
# df = df.drop(['syllable_per_line', 'novel_word_proportion', 'end_pairs_variation','rhymes_per_line'], axis=1)
# df.fillna(0,inplace=True)
# df_111 = df.head(100)

# df.to_csv('data_model.csv',index=False)

# # df = pd.read_csv('data_model.csv')

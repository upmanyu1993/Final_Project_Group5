import pandas as pd
# import syllapy
from nltk.tokenize import sent_tokenize
import numpy as np
# import pronouncing
df = pd.read_csv('spotify_songs2.csv')


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
    return novel_word_props
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

def rhyme_density(lyrics):
    lines = lyrics.split('\n')
    total_syllables = 0
    rhymed_syllables = 0

    for line in lines:
        words = line.split()
        syllables = [syllapy.count(w) for w in words]
        total_syllables += sum(syllables)

        # Check for rhymes and count their syllables
        for i, word in enumerate(words):
            if any(word in pronouncing.rhymes(w) for w in words[:i] + words[i+1:]):
                rhymed_syllables += syllables[i]

    return rhymed_syllables / total_syllables if total_syllables > 0 else 0

def end_pairs_per_line(lyrics):
    lines = lyrics.split('\n')
    total_lines = len(lines)
    end_rhyme_count = 0

    for i in range(total_lines - 1):
        last_word_current = lines[i].split()[-1]
        last_word_next = lines[i + 1].split()[-1]

        if last_word_next in pronouncing.rhymes(last_word_current):
            end_rhyme_count += 1

    return (end_rhyme_count / total_lines) * 100 if total_lines > 0 else 0

def end_pairs_variation(lyrics):
    lines = lyrics.split('\n')
    total_couplets = len(lines) - 1
    grown, shrunk, even = 0, 0, 0

    for i in range(total_couplets):
        syllables_current = syllapy.count(lines[i].split()[-1])
        syllables_next = syllapy.count(lines[i + 1].split()[-1])

        if syllables_next > syllables_current * 1.15:  # 15% longer
            grown += 1
        elif syllables_next < syllables_current * 0.85:  # 15% shorter
            shrunk += 1
        else:
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
        last_word_current = lines[i].split()[-1]
        last_word_next = lines[i + 1].split()[-1]

        if last_word_next in pronouncing.rhymes(last_word_current):
            total_score += phonetic_similarity(last_word_current, last_word_next)
            count += 1

    return total_score / count if count > 0 else 0

def average_end_syl_score(lyrics):
    lines = lyrics.split('\n')
    total_score = 0
    total_syllables = 0

    for i in range(len(lines) - 1):
        last_word_current = lines[i].split()[-1]
        last_word_next = lines[i + 1].split()[-1]

        if last_word_next in pronouncing.rhymes(last_word_current):
            score = phonetic_similarity(last_word_current, last_word_next)
            syllables = syllapy.count(last_word_current) + syllapy.count(last_word_next)
            total_score += score
            total_syllables += syllables

    return total_score / total_syllables if total_syllables > 0 else 0

def count_rhyme_lengths(lyrics):
    lines = lyrics.split('\n')
    rhyme_lengths = {1: 0, 2: 0, 3: 0, 4: 0, 'longs': 0}

    for line in lines:
        words = line.split()
        for word in words:
            rhymes = pronouncing.rhymes(word)
            for rhyme in rhymes:
                syllable_count = syllapy.count(rhyme)
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
    if total_rhymes == 0:  # Avoid division by zero
        return {length: 0 for length in rhyme_lengths}

    # Convert counts to percentages
    rhyme_lengths_percent = {length: (count / total_rhymes) * 100 for length, count in rhyme_lengths.items()}
    return rhyme_lengths_percent


df['lyrics'] = df['lyrics'].apply(lambda x: sent_tok(x))
df['syllable_per_line'] = df['lyrics'].apply(lambda x: syllables_per_line(x))
df['syllable_per_word'] = df['lyrics'].apply(lambda x: syllables_per_word(x))
df['syllable_var'] = df['lyrics'].apply(lambda x: syllable_variation(x))
df['novel_word_proportion'] = df['lyrics'].apply(lambda x: novel_word_proportion(x))
df['rhymes_per_line'] = df['lyrics'].apply(lambda x: rhymes_per_line(x))
df['rhymes_per_syllable'] = df['lyrics'].apply(lambda x: rhymes_per_syllable(x))
df['rhyme_density'] = df['lyrics'].apply(lambda x: rhyme_density(x))
df['end_pairs_per_line'] = df['lyrics'].apply(lambda x: end_pairs_per_line(x))
df['end_pairs_variation'] = df['lyrics'].apply(lambda x: end_pairs_variation(x))
df['average_end_score'] = df['lyrics'].apply(lambda x: average_end_score(x))
df['average_end_syl_score'] = df['lyrics'].apply(lambda x: average_end_syl_score(x))
df['count_rhyme_lengths'] = df['lyrics'].apply(lambda x: count_rhyme_lengths(x))

import prosodic as p
import concurrent.futures

# Configure prosodic for non-verbose mode
p.config['print_to_screen'] = 0

def classify_meter(stress_pattern):
    meters = {
        'iambic': '01',
        'trochaic': '10',
        'spondaic': '11',
        'anapestic': '001',
        'dactylic': '100',
        'amphibrachic': '010',
        'pyrrhic': '00'
    }

    for meter, pattern in meters.items():
        if pattern in stress_pattern:
            return meter
    return 'unknown'

def process_line(line, stress_str2int):
    stress_pattern = ''.join([str(stress_str2int[s]) for s in line.str_stress()])
    return classify_meter(stress_pattern)

def meter_percentages(text, timeout=5):  # Set the timeout in seconds
    pText = p.Text(text)
    pText.parse()
    meter_counts = {meter: 0 for meter in ['iambic', 'trochaic', 'spondaic', 'anapestic', 'dactylic', 'amphibrachic', 'pyrrhic', 'unknown']}

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for line in pText.lines():
            futures.append(executor.submit(process_line, line, line.stress_str2int))

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            try:
                meter = future.result()
                meter_counts[meter] += 1
            except concurrent.futures.TimeoutError:
                print("Processing line timed out")
                meter_counts['unknown'] += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                meter_counts['unknown'] += 1

    total_lines = sum(meter_counts.values())
    meter_percentages = {meter: (count / total_lines) * 100 for meter, count in meter_counts.items()} if total_lines > 0 else meter_counts
    
    return meter_percentages
# print (df.head())
df.index=range(len(df))


from tqdm import tqdm
for i in tqdm(range(len(df))):
    # print (i)
    # i=55
    meter = meter_percentages(df.loc[i,'lyrics'])
    df.at[i,'meter_analysis_iambic'] = meter['iambic']
    df.at[i,'meter_analysis_trochaic'] = meter['trochaic']
    df.at[i,'meter_analysis_spondaic'] = meter['spondaic']
    df.at[i,'meter_analysis_anapestic'] = meter['anapestic']
    df.at[i,'meter_analysis_dactylic'] = meter['dactylic']
    df.at[i,'meter_analysis_amphibrachic'] = meter['amphibrachic']
    df.at[i,'meter_analysis_pyrrhic'] = meter['pyrrhic']    
    df.at[i,'meter_analysis_unknown'] = meter['unknown']    
df.to_csv('/content/drive/MyDrive/spotify_songs3.csv',index=False)


df11 = pd.DataFrame(np.concatenate([df.iloc[:,12:24], df.iloc[:,26:]],axis=1), columns = np.concatenate([df.iloc[:,12:24].columns, df.iloc[:,26:].columns]))
df11['track_popularity'] = df['track_popularity'].tolist()
df = df11.copy()

def expand_list_to_columns(data, max_length):
    # Ensure the data is a list and pad it if necessary
    data = list(data) if data else []
    data += [None] * (max_length - len(data))
    return data[:max_length]
# Maximum length you expect in the lists/tuples

df['syllable_per_line'] = df['syllable_per_line'].apply(lambda x: x.replace(']','').replace('[','').split(','))
df['rhymes_per_line'] = df['rhymes_per_line'].apply(lambda x: x.replace(']','').replace('[','').split(','))
df['novel_word_proportion'] = df['novel_word_proportion'].apply(lambda x: x.replace(']','').replace('[','').split(','))
df['end_pairs_variation'] = df['end_pairs_variation'].apply(lambda x: x.replace(')','').replace('(','').split(','))
max_length = max(df['syllable_per_line'].apply(lambda x: len(x)))
# Apply the function to each column and create new columns
for col in ['syllable_per_line', 'novel_word_proportion', 'end_pairs_variation','rhymes_per_line']:
    expanded_cols = df[col].apply(lambda x: expand_list_to_columns(x, max_length))
    expanded_col_names = [f'{col}_{i+1}' for i in range(max_length)]
    df[expanded_col_names] = pd.DataFrame(expanded_cols.tolist(), index=df.index)

# Drop the original columns if they are no longer needed
df = df.drop(['syllable_per_line', 'novel_word_proportion', 'end_pairs_variation','rhymes_per_line'], axis=1)
df.fillna(0,inplace=True)
df_111 = df.head(100)

df.to_csv('data_model.csv',index=False)

# # df = pd.read_csv('data_model.csv')

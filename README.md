# Final_Project_Group5: Lyric Generation & Analysis
Welcome to the **Final Project Group 5** repository. This project leverages GPT-2 to generate song lyrics, implements a custom algorithm to summarize lyrics, and extracts keywords to enhance lyrical analysis.

## Project Overview

This repository contains Python scripts for:
- **Generating Lyrics:** Using GPT-2 to create original song lyrics.
- **Summarizing Lyrics:** Applying a custom algorithm to condense lyrics into their key themes.
- **Extracting Keywords:** Identifying the most important words in a set of lyrics for further analysis.

[The link to download all the models](https://storage.googleapis.com/nlp-final-project-group05/Models.zip). Please put all model files in the folder <code> /Code </code>. 

Here’s a glimpse of the app in action:

<img width="288" alt="App Running Example 1" src="https://github.com/user-attachments/assets/df92180b-c553-4c91-807c-6cd7ca3a38d4">
<img width="288" alt="App Running Example 2" src="https://github.com/user-attachments/assets/ba882601-7b99-4130-915c-013bfb19b11f">
<img width="288" alt="App Running Example 3" src="https://github.com/user-attachments/assets/e7a44d51-5ebe-45d7-be6b-c10f808c4028">
<img width="288" alt="App Running Example 4" src="https://github.com/user-attachments/assets/ee22d96f-2b8e-455c-903b-e74e47ac640b">


### Lyric Generation

To generate lyrics using GPT-2, run the following script:

python Lyrics_generator_byGPT2_fin.py

This script will create 10 training datasets and 10 evaluation datasets. The GPT-2 model will then be trained using these datasets to generate lyrics. There is no specific order to run scripts; you can run this script independently.

### Lyrics Summarization
To use the lyrics summarization and keyword extraction features, run the Lyrics_Summarization.py file:
python Lyrics_Summarization.py
This script will provide a lyric summarizer and keyword extractor. There is no specific order to run scripts; you can run this script independently.

For App,
### Lyric Generation
The sample.py file contains the lyric generation functionality for the app. The generated lyrics will be seamlessly integrated into the main app.

### Summarization and Keyword Extraction
The lyrics_analysis.py file provides functions for lyric summarization and keyword extraction. These functions enhance the analysis capabilities of the main app, giving users deeper insights into the lyrics.

### Further Analysis
Explore additional analysis in the pages/1_Further_Analysis.py file, which contains Markdown content about further analysis and features of the app.

The app utilizes the lyrics_analysis.py and sample.py scripts to offer a seamless experience for users.
Main App: To run the main app, execute the Lyric_Genie.py file - This will launch the Streamlit app, providing you with a user interface to interact with the lyric generation, summarization, and keyword extraction features.

Lyrics feature plus lyrics sample:  This is the regression model which uses cnn and does the preprocessing

Lyrics gatherer: It gives the embedding to the songs lyrics

feature creation: It creates new features on the basis of lyrics

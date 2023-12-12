# Final_Project_Group5
This repository contains Python scripts for generating lyrics using GPT-2 and summarizing lyrics using a custom lyric summarization algorithm. Additionally, it provides functionality for extracting keywords from lyrics.


[The link to download all the models](https://storage.googleapis.com/nlp-final-project-group05/Models.zip). Please put all model files in the folder <code> /Code </code>. 

python Lyrics_generator_byGPT2_fin.py
This script will create 10 training datasets and 10 evaluation datasets. The GPT-2 model will then be trained using these datasets to generate lyrics. There is no specific order to run scripts; you can run this script independently.

Lyrics Summarization
To use the lyrics summarization and keyword extraction features, run the Lyrics_Summarization.py file:
python Lyrics_Summarization.py
This script will provide a lyric summarizer and keyword extractor. There is no specific order to run scripts; you can run this script independently.

For App,
Lyric Generation
The sample.py file contains the lyric generation functionality for the app. The generated lyrics will be seamlessly integrated into the main app.

Summarization and Keyword Extraction
The lyrics_analysis.py file provides functions for lyric summarization and keyword extraction. These functions enhance the analysis capabilities of the main app, giving users deeper insights into the lyrics.

Further Analysis
Explore additional analysis in the pages/1_Further_Analysis.py file, which contains Markdown content about further analysis and features of the app.

The app utilizes the lyrics_analysis.py and sample.py scripts to offer a seamless experience for users.
Main App: To run the main app, execute the Lyric_Genie.py file - This will launch the Streamlit app, providing you with a user interface to interact with the lyric generation, summarization, and keyword extraction features.

Lyrics feature plus lyrics sample:  This is the regression model which uses cnn and does the preprocessing

Lyrics gatherer: It gives the embedding to the songs lyrics

feature creation: It creates new features on the basis of lyrics
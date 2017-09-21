# PersonalityChatbot
A chatbot whose responses mimic personality of popular TV series characters.

## Data Sources-
- [F.R.I.E.N.D.S. transcripts](https://fangj.github.io/friends/)
- [The Big Bang Theory transcripts](https://bigbangtrans.wordpress.com/about/)

## Methodology-
- Scraped all the episodes of all seasons of the TV series using [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/)
- Converted the script text to lowercase, removed rare alphanumeric characters
- Mapped alphanumeric characters to numbers
- A multiclass classficiation problem- each class is an alphanumeric character
- Converted the script text into short sequences
- Built a 3 Layer LSTM RNN, trained it on the sequences (as X) and the next alphanumeric character (as y)
- Given a seed input (a short sequence) from the script, the model predicts one alphanumeric character at a time
*algorithm credits- fchollet*

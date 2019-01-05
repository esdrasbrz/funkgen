"""
Crawler module to scrap funk lyrics from letras.mus.br
"""

import requests
import random
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from progress.bar import Bar
import re

BASE_URL = 'https://www.letras.mus.br/mais-acessadas/funk/'

def get_lyrics(html):
    s = BeautifulSoup(html, 'html.parser')
    article = s.find('article')

    lyrics = re.sub(r'</?article>', '', str(article))
    lyrics = re.sub(r'<.+?>', '\n', lyrics)

    sentences = lyrics.split('\n')
    # filter sentences with less than 3 words
    sentences = [s for s in sentences if len(s.split()) >= 3]

    return sentences
    
def scrap(output_train_file, output_test_file, test_percentage=3, n_songs=1):
    # access base url with top 1000 funks
    html = requests.get(BASE_URL).text
    soup = BeautifulSoup(html, 'html.parser')

    links = [l['href'] for l in soup.find('ol', {'class': 'top-list_mus'}).find_all('a')]

    # limit the number of songs
    if n_songs != -1 and n_songs < len(links):
        links = links[:n_songs]

    # iterate over all links and get the lyrics
    songs = []
    bar = Bar('Scraping', max=len(links))
    for link in links:
        url = urljoin(BASE_URL, link)
        html_lyrics = requests.get(url).text
        lyrics = get_lyrics(html_lyrics)
        songs.append(lyrics)

        bar.next()
    bar.finish()

    random.shuffle(songs) 

    test_split_index = int(test_percentage * len(songs) / 100.)
    test_songs = songs[:test_split_index]
    train_songs = songs[test_split_index:]
    
    train_corpus = ""
    test_corpus = ""
    
    print("Saving dataset...")
    with open(output_train_file, 'w') as fout:
        for s in train_songs:
            seq = '\n'.join(s)
            train_corpus += seq + '\n'

            fout.write(seq + '\n')

    with open(output_test_file, 'w') as fout:
        for s in test_songs:
            seq = '\n'.join(s)
            test_corpus += seq + '\n'

            fout.write(seq + '\n')
    print("Done!")

    return train_corpus, test_corpus

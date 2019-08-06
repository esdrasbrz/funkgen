"""
Crawler module to scrap funk lyrics from letras.mus.br
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from tqdm import tqdm
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
    
def scrap(output_file, n_songs=1):
    # access base url with top 1000 funks
    html = requests.get(BASE_URL).text
    soup = BeautifulSoup(html, 'html.parser')

    links = [l['href'] for l in soup.find('ol', {'class': 'top-list_mus'}).find_all('a')]

    # limit the number of songs
    if n_songs != -1 and n_songs < len(links):
        links = links[:n_songs]

    # iterate over all links and get the lyrics
    sentences = []
    for link in tqdm(links):
        url = urljoin(BASE_URL, link)
        html_lyrics = requests.get(url).text
        lyrics = get_lyrics(html_lyrics)
        sentences.extend(lyrics)

    print("Saving dataset...")
    text = '\n'.join(sentences).lower()
    with open(output_file, 'w') as fout:
        fout.write(text)
    print("Done!")

    return text

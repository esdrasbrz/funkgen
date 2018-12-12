"""
Crawler module to scrap funk lyrics from letras.mus.br
"""

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from progress.bar import Bar
import config

BASE_URL = 'https://www.letras.mus.br/mais-acessadas/funk/'

def get_lyrics(html):
    s = BeautifulSoup(html, 'html.parser')
    article = s.find('article')

    for elem in article.find_all("br"):
        elem.replace_with("\n")

    sentences = article.text.split('\n')
    # filter sentences with less than 3 words
    sentences = [s for s in sentences if len(s.split()) >= 3]

    return sentences
    
def scrap(n_songs=1):
    # access base url with top 1000 funks
    html = requests.get(BASE_URL).text
    soup = BeautifulSoup(html, 'html.parser')

    links = [l['href'] for l in soup.find('ol', {'class': 'top-list_mus'}).find_all('a')]

    # limit the number of songs
    if n_songs != -1 and n_songs < len(links):
        links = links[:n_songs]

    # iterate over all links and get the lyrics
    sentences = []
    bar = Bar('Scraping', max=len(links))
    for link in links:
        url = urljoin(BASE_URL, link)
        html_lyrics = requests.get(url).text
        lyrics = get_lyrics(html_lyrics)
        sentences.extend(lyrics)

        bar.next()
    bar.finish()

    print("Saving dataset...")
    with open(config.OUTPUT_DATASET_FILE, 'w') as fout:
        fout.write('\n'.join(sentences))
    print("Done!")

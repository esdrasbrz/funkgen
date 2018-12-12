"""
Crawler used to generate dataset of Favela Funk
"""

import letrasmus
import config

def main():
    letrasmus.scrap(n_songs=config.NUM_SONGS)

if __name__ == '__main__':
    main()

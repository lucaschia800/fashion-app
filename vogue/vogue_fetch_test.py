import json
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import os
from unidecode import unidecode


def designer_show_to_download_images(designer, show):
    # Replace spaces with - and lowercase
    show = show.replace(' ','-').lower()
    show = unidecode(show)

    # Replace spaces, puncuations, special character, etc. with - and make lowercase
    designer = designer.replace(' ','-').replace('.','-').replace('&','').replace('+','').replace('--','-').lower()
    designer = unidecode(designer)

    # URL of the show
    url = "https://www.vogue.com/fashion-shows/" + show + '/' + designer

    # Make request
    r = requests.get(url)

    # Soupify
    soup = BeautifulSoup(r.content, 'html5lib') # If this line causes an error, run 'pip install html5lib' or install html5lib

    # Load a dict of the json file with the relevent data
    
    try:
        js = str(soup.find_all('script', type='text/javascript')[3])
        js = js.split(' = ')[1]
        js = js.split(';</') [0]
        data = json.loads(js)
        print(data)
   
    except:
        print('could not find js code')
        return None
    

designer_show_to_download_images('acne-studios', 'fall-2025-ready-to-wear')
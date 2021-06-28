from urllib.request import Request, urlopen
import os
import re
import requests

url_base = "http://www.shipspotting.com/gallery/search.php?search_category_1="
categories = [
    276,169,37,160,65,18,168,5,137,140,141,142,143,144,257,293,145,182,202,38,39,55,46,25,291,277,4,205,132,133,134,129,261,294,131,164,147,146,155,13,193,194,195,196,197,198,262,297,199,167,166,30,273,94,220,284,9,217,238,14,239,149,41,181,165,58,265,298,264,171,15,102,173,103,174,104,175,105,176,106,178,127,179,128,180,259,258,296,295,101,172,263,268,148,212,218,275,53,52,51,32,249,282,31,43,184,187,185,188,189,162,42,244,78,23,161,191,33,69,48,80,44,204,283,76,242,57,60,163,19,274,96,20,285,224,223,243,50,17,245,21,190,281,70,100,159,22,83,81,71,77,95,73,97,62,92,49,170,35,278,8,156,24,45,63,10,270,221,59,12,216,280,241,82,64,211,34,61,208,219,210,209
]
photos_url_regex = r'http:\/\/www\.shipspotting\.com\/photos\/[^"]*'


for category in categories:
    # get page plaintext
    print("requesting " + url_base + category.__str__() + "&mod_page_limit=9999")
    request = Request(url_base + category.__str__() + "&mod_page_limit=9999", headers={'User-Agent': 'Mozilla/5.0'})
    page = urlopen(request)
    print("reading and decoding...")
    page_bytes = page.read()
    page_plaintext = page_bytes.decode('iso-8859-1')

    # look for all matches with photos url
    print("Scanning for images...")
    photos = re.findall(photos_url_regex, page_plaintext)
    print(photos.__len__().__str__() + " images found")

    # fetch each photo, into category folder
    print("Saving images to category folder")
    os.mkdir(str(category))
    for photo_url in photos:
        file_path = os.path.join(str(category), photo_url[photo_url.rfind("/") + 1:])
        res = requests.get(photo_url).content
        with open(file_path, "wb") as f:
            f.write(res)

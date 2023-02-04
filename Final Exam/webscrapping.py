# ### Python script to scrape a url can output clean and useful information into a csv file
import csv

import requests
from bs4 import BeautifulSoup

url = "https://www.yelp.ca/biz/pai-northern-thai-kitchen-toronto-5?osq=Restaurants"

response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

restaurant = soup.find("h1", class_="css-1se8maq").text
total_reviews = soup.find("a", class_="css-1m051bw").text

print(f"Restaurant name: {restaurant}")
print(f"Total reviews: {total_reviews}")

reviews = soup.find_all("div", class_="review__09f24__oHr9V border-color--default__09f24__NPAKY")

output = []
review = reviews[0]
for review in reviews:
    name = review.find("a", class_="css-1m051bw").text

    text = review.find("p", class_="comment__09f24__gu0rG css-qgunke").text

    rating = review.find("div", class_="five-stars__09f24__mBKym five-stars--regular__09f24__DgBNj display--inline-block__09f24__fEDiJ border-color--default__09f24__NPAKY")
    rating = int(rating.attrs["aria-label"][0])

    output.append([name, text, rating])

header = ["Reviewer", "Review_text", "Rating"]

with open('output.csv', 'w', encoding='UTF8', newline='') as f:
    writer = csv.writer(f)

    # write the header
    writer.writerow(header)

    # write multiple rows
    writer.writerows(output)
    print("Output written to output.csv")
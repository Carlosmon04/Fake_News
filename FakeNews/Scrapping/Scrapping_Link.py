from bs4 import BeautifulSoup
import requests


URL="https://www.reddit.com/r/NoStupidQuestions/comments/1h3d2tu/why_are_there_so_many_attractive_people_at_the/"

page= requests.get(URL)

soup= BeautifulSoup(page.text,'html.parser')
# print(soup.prettify())

titulo = soup.find_all('h1',class_="font-semibold")
Body= soup.find_all('div',class_="text-neutral-content")


print("TITULO")
print("")
if titulo:
    titulo_texto= titulo[0].get_text(strip=True)
    print(titulo_texto)
else:
    print("No hay Titulos")
    print(" ")
# print(Body)
print(" ")
print("CUERPO")
print("")
for div in Body:
    parrafos= div.find_all('p')

    for parrafo in parrafos:
        texto=parrafo.get_text(strip=True)
        if texto:
            print(texto)


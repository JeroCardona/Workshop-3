from django.shortcuts import render
from django.http import HttpResponse

from movie.models import Movie

import matplotlib.pyplot as plt
import matplotlib
import io
import urllib, base64
from django.shortcuts import render
from dotenv import load_dotenv, find_dotenv
import os
import json
from openai import OpenAI
import numpy as np

_ = load_dotenv('C:/Users/USUARIO/Documents/U/P1/Taller3/Workshop-3/openAI.env')
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('openAI_api_key'),
)

with open('C:/Users/USUARIO/Documents/U/P1/Taller3/Workshop-3/movie_descriptions_embeddings.json', 'r') as file:

    file_content = file.read()
    movie_embs = json.loads(file_content)

#Esta función devuelve una representación numérica (embedding) de un texto, en este caso
#la descripción de las películas
    
def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def recommendations(request):
    recommendTerm = request.GET.get('recommendMovie') # GET se usa para solicitar recursos de un servidor
    if recommendTerm:
        emb_req = get_embedding(recommendTerm)
        sim = []
        for i in range(len(movie_embs)):
            sim.append(cosine_similarity(emb_req,movie_embs[i]['embedding']))
        sim = np.array(sim)
        idx = np.argmax(sim)
        recommendMovie = Movie.objects.filter(title__icontains=movie_embs[idx]['title'])
    else:
        recommendMovie = Movie.objects.all()
    return render(request, 'recommendations.html', {'recommendTerm':recommendTerm, 'recommendMovie':recommendMovie})
import json
from sentence_transformers import SentenceTransformer
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

model = None
def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

# def load_catalog():

#     with open("D:\SHL_assessment\data\shl_product_catalog.json",encoding='utf8') as f:

#         data=json.load(f)

#     return data

def load_catalog():
    path = Path("data/shl_product_catalog.json")

    with open(path, encoding="utf8") as f:
        return json.load(f)

def build_docs(data):

    docs=[]

    for item in data:

        text=f"""

        {item['name']}

        {' '.join(item['keys'])}

        {' '.join(item['job_levels'])}

        {item['description']}

        {item['duration']}

        """

        docs.append(text)

    return docs


def create_embeddings():
    model = get_model()
    catalog=load_catalog()
    docs=build_docs(catalog)
    emb=model.encode(docs)

    return emb,catalog
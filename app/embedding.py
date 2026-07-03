import json
from sentence_transformers import SentenceTransformer
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

model = None
def get_model():
    global model
    if model is None:
        model = SentenceTransformer("all-MiniLM-L6-v2")
    return model

def load_catalog():

    with open("D:\SHL_assessment\data\shl_product_catalog.json",encoding='utf8') as f:

        data=json.load(f)

    return data


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

    catalog=load_catalog()

    docs=build_docs(catalog)

    emb=model.encode(
            docs
    )

    return emb,catalog
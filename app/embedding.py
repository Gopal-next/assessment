import json
import warnings
from pathlib import Path
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv()

warnings.filterwarnings("ignore", category=FutureWarning)

model = None

def get_model():

    global model

    if model is None:

        model = GoogleGenerativeAIEmbeddings(

            model="models/gemini-embedding-2"

        )

    return model
#local
# def load_catalog():

#     with open("D:\SHL_assessment\data\shl_product_catalog.json",encoding='utf8') as f:

#         data=json.load(f)

#     return data
def load_catalog():

    with open(BASE_DIR / "data" / "shl_product_catalog.json",encoding='utf8') as f:

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
    model = get_model()
    catalog=load_catalog()
    docs=build_docs(catalog)
    emb=model.embed_documents(docs)

    return emb,catalog



# def load_catalog():
#     path = Path("data/shl_product_catalog.json")

#     with open(path, encoding="utf8") as f:
#         return json.load(f)
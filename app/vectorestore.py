# import faiss
import numpy as np
from app.embedding import create_embeddings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
# emb,catalog=create_embeddings()
# dim=emb.shape[1]
# index=faiss.IndexFlatIP(
#     dim
# )

# index.add(
#         np.array(emb)
# )

# def search(query_embedding,k=5):

#     scores,idx=index.search(

#             query_embedding,

#             k

#     )

#     return idx


import faiss
import json

# index = faiss.read_index(

#     "D:\SHL_assessment\data\index.faiss"

# )

# with open(

#     "D:\SHL_assessment\data\shl_product_catalog.json",

#     encoding="utf8"

# ) as f:

#     catalog = json.load(f)


index = faiss.read_index(
    str(BASE_DIR / "data" / "index.faiss")
)



with open(
    BASE_DIR / "data" / "shl_product_catalog.json",
    encoding="utf8"
) as f:
    catalog = json.load(f)
    
def search(query_embedding, k=5):
    D, I = index.search(query_embedding, k)
    return I

    return idx
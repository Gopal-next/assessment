import faiss
import numpy as np
from app.embedding import create_embeddings

emb,catalog=create_embeddings()
dim=emb.shape[1]
index=faiss.IndexFlatIP(
    dim
)

index.add(
        np.array(emb)
)

def search(query_embedding,k=5):

    scores,idx=index.search(

            query_embedding,

            k

    )

    return idx
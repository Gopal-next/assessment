import json
import faiss
import numpy as np
import time

from embedding import get_model

model = get_model()

with open(
    r"D:\\SHL_assessment\\data\\shl_product_catalog.json",
    encoding="utf8"
) as f:
    catalog = json.load(f)

docs = [
    f"{item['name']} {item.get('description','')}"
    for item in catalog
]

embeddings = []

batch_size = 20

import time

for i in range(0, len(docs), 20):

    batch = docs[i:i+20]

    while True:

        try:

            emb = model.embed_documents(batch)

            embeddings.extend(emb)

            break

        except Exception as e:

            if "RESOURCE_EXHAUSTED" in str(e):

                print("Quota hit. Waiting 30 sec...")

                time.sleep(64)

            else:

                raise e

    processed = i + len(batch)

    print(f"Processed {processed}/{len(docs)}")

    print(
        "Last item:",
        " ".join(batch[-1].split()[-30:])
    )

embeddings = np.array(
    embeddings,
    dtype=np.float32
)

index = faiss.IndexFlatL2(
    embeddings.shape[1]
)

index.add(embeddings)

index = faiss.write_index(
    index,
    r"D:\\SHL_assessment\\data\\index.faiss"
)

print("FAISS index saved")
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3")


def get_embedding(text: str):
    return model.encode([text], return_dense=True)["dense_vecs"][0].tolist()

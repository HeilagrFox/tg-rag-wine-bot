from autogen_core.tools import FunctionTool
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch
from qdrant_client.http.models import Document
from ..embeddings import get_embedding
from loguru import logger
from ..config import load_config

config = load_config()
QDRANT_INFO = config.get("QDRANT", {})

qdrant_client = QdrantClient(
    url=QDRANT_INFO.get("URL", ""),
    api_key=QDRANT_INFO.get("API_KEY", ""),
)


async def search_wines_by_query(query: str, limit: int = 3) -> str:

    if limit > 10:
        limit = 10

    try:
        dense_query = get_embedding(query)
    except Exception as e:
        logger.error(f"Ошибка эмбеддинга: {e}")
        return "Не удалось обработать запрос."

    result = qdrant_client.query_points(
        collection_name="wines",
        prefetch=[
            Prefetch(query=dense_query, using="dense"),
            Prefetch(query=Document(text=query, model="qdrant/bm25"), using="bm25"),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    hits = result.points
    if not hits:
        return "Ничего не найдено по вашему запросу."

    results = []
    for point in hits:
        p = point.payload
        name = p.get("Name", "")
        price = p.get("Price", "")
        country = p.get("Country", "")
        color = p.get("Color", "")
        acidity = p.get("Acidity", "")
        text = p.get("text", "")
        volume = p.get("Volume", "")
        line = f"{name} | {country} | {color} | {acidity} | {price} | {volume} \n Описание: {text}"
        results.append(line)

    return "\n\n".join(results)


description = """
    Выполняет семантический (гибридный) поиск по описанию, вопросу или названию.
    Используется для общих запросов: "что подходит к рыбе?", "расскажи про Шабли", "что ты знаешь про регион - ...", "какие блюда подходят к ...".
    Используй для любых текстовых запросов про вино: названия, описания, пары с едой, регионы, рекомендации, сравнения. "
    Args:
        query (str): Текстовый запрос.
        limit (int): Максимум результатов (по умолчанию 3, максимум 10).
    
    Returns:
        str: Релевантные вина или сообщение об ошибке.
    """
search_wines_query_tool = FunctionTool(search_wines_by_query, description=description)

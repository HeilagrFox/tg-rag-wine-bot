from typing import Optional
from qdrant_client.http.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    MatchText,
)
from autogen_core.tools import FunctionTool
from qdrant_client import QdrantClient
from ..config import load_config

config = load_config()
QDRANT_INFO = config.get("QDRANT", {})

qdrant_client = QdrantClient(
    url=QDRANT_INFO.get("URL", ""),
    api_key=QDRANT_INFO.get("API_KEY", ""),
)


def search_wines_by_attributes(
    color: Optional[str] = None,
    country: Optional[str] = None,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    acidity: Optional[str] = None,
    limit: int = 3,
) -> str:
    """
    Ищет вина по структурированным атрибутам (цена, цвет, страна, кислотность).
    Подходит для фильтрации каталога.

    Args:
        color (str, optional): Цвет вина. Пример: "Red", "White".
        country (str, optional): Страна на английском. Пример: "France", "USA".
        min_price (float, optional): Минимальная цена.
        max_price (float, optional): Максимальная цена.
        acidity (str, optional): Кислотность. Пример: "Dry", "Sweet", "Semi-Dry".
        limit (int): Максимум результатов (по умолчанию 3, максимум 15).

    Returns:
        str: Список найденных вин или сообщение, что ничего не найдено.
    """
    if limit > 30:
        limit = 30

    conditions = []

    if color:
        conditions.append(
            FieldCondition(key="Color", match=MatchValue(value=color.lower()))
        )
    if country:
        conditions.append(FieldCondition(key="Country", match=MatchText(value=country)))
    if acidity:
        conditions.append(
            FieldCondition(key="Acidity", match=MatchValue(value=acidity))
        )

    if min_price is not None or max_price is not None:
        price_range = {}
        if min_price is not None:
            price_range["gte"] = min_price
        if max_price is not None:
            price_range["lte"] = max_price
        conditions.append(FieldCondition(key="Price", range=Range(**price_range)))

    if not conditions:
        return "Укажите хотя бы один параметр: цвет, страну, цену или кислотность."

    filter_obj = Filter(must=conditions)
    hits, _ = qdrant_client.scroll(
        collection_name="wines",
        scroll_filter=filter_obj,
        limit=limit,
        with_payload=True,
    )

    if not hits:
        return "Ничего не найдено по заданным критериям."

    results = []
    for point in hits:
        p = point.payload
        name = p.get("Name", "")
        price = p.get("Price", "")
        country_val = p.get("Country", "")
        color_val = p.get("Color", "")
        acidity_val = p.get("Acidity", "")
        text = p.get("text", "")
        volume = p.get("Volume", "")
        line = f"{name} | {country_val} | {color_val} | {acidity_val} | {price} | {volume} \n Описание: {text}"
        results.append(line)

    return "\n\n".join(results)


description = """
    Ищет вина по структурированным атрибутам (цена, цвет, страна, кислотность).
    Подходит для фильтрации каталога. 
    Используй, когда пользователь хочет найти вина по ПАРАМЕТРАМ: цвет, страна, цена, кислотность. 
    НИКОГДА НЕ используй для общих вопросов про вино, блюда или регионы."
    Args:
        color (str, optional): Цвет вина. Есть всего один вариант: "Красное" 
        country (str, optional): Страна производства (например: "USA", "Fr", "It"). Страна всегда на английском.
        min_price (float, optional): Минимальная цена (больше чем). Максимальная цена не обязательна, валюта в рублях
        max_price (float, optional): Максимальная цена (меньше чем). Минимальная цена не обязательна, валюта в рублях
        acidity (str, optional): кислотность вина. Принимает значения: "Сладкое","Сухое","Полусладкое","Полусухое" 
        limit (int, optional): Максимальное количество результатов (по умолчанию 3, максимум используй 30). Лимит не может быть использован сам по себе
    Returns:
        str: Отформатированный список найденных вин или сообщение об ошибке.
    """
search_wines_attributes_tool = FunctionTool(
    search_wines_by_attributes, description=description
)

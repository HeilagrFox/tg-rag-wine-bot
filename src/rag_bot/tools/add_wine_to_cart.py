from typing import List, Dict, Any
from autogen_core.tools import FunctionTool


def create_add_to_cart_tool(user_id: int, shopping_carts: dict):
    async def add_to_cart(wine_name: str, wine_details: str = "") -> str:

        if user_id not in shopping_carts:
            shopping_carts[user_id] = []

        wine_entry = {"name": wine_name.strip(), "details": wine_details.strip()}
        shopping_carts[user_id].append(wine_entry)
        return f"Вино '{wine_name}' добавлено в вашу корзину!"

    return FunctionTool(
        func=add_to_cart,
        description="Добавить указанное вино в корзину пользователя. Используй только когда пользователь просит это явно.  Args: wine_name (str): Название вина. ; wine_details (str, optional): Дополнительная информация (регион, сорт и т.д.).",
    )

import logging
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.agents import AssistantAgent
from .tools.search_wines_by_query import search_wines_query_tool
from .tools.search_wines_by_attributes import search_wines_attributes_tool
from .tools.add_wine_to_cart import create_add_to_cart_tool
from autogen_core.model_context import BufferedChatCompletionContext
from .config import load_config
from loguru import logger

config = load_config()
MODEL_INFO = config.get("MODEL_CLIENT", {})
TG_TOKEN = config.get("TG_BOT_TOKEN", "")

model_client = OpenAIChatCompletionClient(
    model=MODEL_INFO.get("MODEL", ""),
    api_key=MODEL_INFO.get("API_KEY", ""),
    base_url=MODEL_INFO.get("BASE_URL", ""),
    model_info={
        "family": MODEL_INFO.get("FAMILY", ""),
        "context_window": 32768,
        "max_prompt_tokens": 30000,
        "max_completion_tokens": 4192,
        "vision": False,
        "function_calling": True,
        "json_output": True,
        "streaming": False,
    },
)
user_agents = {}
shopping_carts = {}


def get_user_agent(user_id: int) -> AssistantAgent:
    if user_id not in user_agents:
        # Создаём тулзу, привязанную к user_id
        add_to_cart_tool = create_add_to_cart_tool(user_id, shopping_carts)

        agent = AssistantAgent(
            name="WineAssistant",
            model_client=model_client,
            system_message=(
                "Вы — эксперт по винам. Отвечайте кратко и точно. "
                "Когда вы получаете результаты поиска, НЕ копируйте их дословно, выделите ключевые детали"
                """Вам доступны для поиска информации два инструмента:

                1. **search_wines_attributes_tool** — используется ТОЛЬКО если пользователь явно указывает **один или несколько фильтров**: цвет, страна, цена (мин/макс), кислотность.
                - Примеры: "Красные вина до 2000 руб", "сухие белые вина из Франции", "вино от 1000 до 1500 рублей".
                - Если запрос не содержит **конкретных значений параметров**, НЕ используйте этот инструмент.
                - Если параметры есть, но запрос также содержит общий вопрос ("что подходит к утке?"), — **НЕ используйте** этот инструмент.

                2. **search_wines_query_tool** — используется во ВСЕХ остальных случаях:
                - Вопросы про еду ("к вуоке?"), регионы ("расскажи про Бордо"), типы вин ("что такое Пино Нуар?"), рекомендации, сравнения, описания.
                - Даже если в запросе есть слово "Франция" или "белое", но без **чёткого намерения фильтровать каталог** — используйте search_wines_query_tool. Потому что этоь инструмент для семантического поиска

                """
                "Если пользователь просит добавить вино в корзину — используйте соответствующий инструмент."
            ),
            model_client_stream=False,
            tools=[
                search_wines_query_tool,
                search_wines_attributes_tool,
                add_to_cart_tool,
            ], 
            max_tool_iterations=10,
            model_context=BufferedChatCompletionContext(buffer_size=5),
        )
        user_agents[user_id] = agent
    return user_agents[user_id]


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Привет! Я винный ассистент. Спросите о винах!")


async def show_cart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    cart = shopping_carts.get(user_id, [])

    if not cart:
        await update.message.reply_text("🛒 Ваша корзина пуста.")
    else:
        items = []
        for item in cart:
            name = item.get("name", "Неизвестное вино")
            details = item.get("details", "").strip()
            if details:
                items.append(f"• {name} — {details}")
            else:
                items.append(f"• {name}")

        cart_text = "\n".join(items)
        await update.message.reply_text(f"🍷 Ваша корзина:\n\n{cart_text}")


async def clear_cart(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    if user_id in shopping_carts:
        del shopping_carts[user_id]
        await update.message.reply_text("Ваша корзина успешно очищена!")
    else:
        await update.message.reply_text("🛒 Ваша корзина и так пуста.")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    user_message = update.message.text

    thinking_msg = await update.message.reply_text("Сомелье в раздумье...")
    agent = get_user_agent(user_id)
    response = await agent.run(task=user_message)

    last_message = response.messages[-1].content
    if len(last_message) > 4096:
        last_message = last_message[:4093] + "..."

    await thinking_msg.edit_text(last_message)


def main():
    application = Application.builder().token(TG_TOKEN).build()
    logger.success("Telegram подключён. Запуск polling...")
    application.add_handler(CommandHandler("start", start))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )
    application.add_handler(CommandHandler("show_cart", show_cart))
    application.add_handler(CommandHandler("clear_cart", clear_cart))
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()

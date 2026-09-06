"""Private owner-only model controls. Callback payloads never contain secrets."""

import logging

from aiogram import F
from aiogram.filters import Command
from aiogram.types import CallbackQuery, InlineKeyboardButton, InlineKeyboardMarkup, Message

from app.services.checkpoints import CheckpointError
from app.services.llm import LLMError
from app.services.model_preferences import model_key

logger = logging.getLogger(__name__)


async def menu(preferences, page=None):
    try:
        current = await preferences.description()
    except LLMError:
        current = "Выбранная модель отключена. Выбери другую."
    rows = [[InlineKeyboardButton(text="Бесплатно · автоподбор", callback_data="model:auto")]]
    if page is None:
        rows.append(
            [InlineKeyboardButton(text="Выбрать бесплатную модель", callback_data="model:page:0")]
        )
    else:
        models = await preferences.choices()
        page = min(max(0, page), max(0, (len(models) - 1) // 6))
        for model in models[page * 6 : (page + 1) * 6]:
            rows.append(
                [InlineKeyboardButton(text=model, callback_data=f"model:free:{model_key(model)}")]
            )
        nav = []
        if page:
            nav.append(InlineKeyboardButton(text="←", callback_data=f"model:page:{page - 1}"))
        if (page + 1) * 6 < len(models):
            nav.append(InlineKeyboardButton(text="→", callback_data=f"model:page:{page + 1}"))
        if nav:
            rows.append(nav)
    if preferences.settings.openrouter_allow_paid:
        for model in preferences.paid_models():
            rows.append(
                [
                    InlineKeyboardButton(
                        text=f"Платно · {model}", callback_data=f"model:paid:{model_key(model)}"
                    )
                ]
            )
    text = (
        f"Модель для новых записей\n{current}\n\n"
        "Выбор общий для лички и канала. Уже принятые записи сохраняют прежнюю модель.\n"
        "Бесплатный режим: без платных резервных моделей и дополнительных платных проверок. "
        "Квоты и доступность бесплатных моделей могут меняться."
    )
    if not preferences.settings.openrouter_allow_paid:
        text += "\nПлатные модели выключены в настройках сервера."
    return text, InlineKeyboardMarkup(inline_keyboard=rows)


def register_settings_menu(router, settings, preferences):
    @router.message(F.chat.type == "private", Command("settings", "models"))
    async def show(message: Message):
        if not message.from_user or message.from_user.id != settings.owner_telegram_id:
            return
        if preferences is None:
            await message.answer("Меню моделей доступно при TEXT_PROVIDER=openrouter.")
            return
        try:
            text, keyboard = await menu(preferences)
            await message.answer(text, reply_markup=keyboard, parse_mode=None)
        except (CheckpointError, LLMError):
            logger.exception("Could not load model preferences")
            await message.answer(
                "Не удалось загрузить настройки. Проверь подключение к базе и повтори /settings."
            )

    @router.callback_query(F.data.startswith("model:"))
    async def choose(query: CallbackQuery):
        if (
            query.from_user.id != settings.owner_telegram_id
            or not isinstance(query.message, Message)
            or query.message.chat.type != "private"
            or query.message.chat.id != settings.owner_telegram_id
        ):
            await query.answer("Настройки доступны только владельцу в личке.", show_alert=True)
            return
        await query.answer()
        if preferences is None:
            return
        try:
            parts = (query.data or "").split(":")
            page = None
            if parts == ["model", "auto"]:
                await preferences.select("free", "openrouter/free")
            elif len(parts) == 3 and parts[1] == "page":
                page = int(parts[2])
            elif len(parts) == 3 and parts[1] in ("free", "paid"):
                models = (
                    await preferences.choices() if parts[1] == "free" else preferences.paid_models()
                )
                model = next((m for m in models if model_key(m) == parts[2]), None)
                if model is None:
                    raise LLMError("Модель больше недоступна. Обнови /settings.")
                await preferences.select(parts[1], model)
            else:
                raise LLMError("Обнови меню командой /settings.")
            text, keyboard = await menu(preferences, page)
            await query.message.answer(text, reply_markup=keyboard, parse_mode=None)
        except (LLMError, CheckpointError, ValueError) as error:
            logger.warning("Model selection failed: %s", type(error).__name__)
            text = (
                str(error)
                if isinstance(error, LLMError)
                else "Не удалось сохранить настройки. Проверь базу и повтори /settings."
            )
            await query.message.answer(text, parse_mode=None)

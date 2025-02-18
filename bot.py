import logging
import os
import torch
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          AutoModelForSequenceClassification, set_seed)
from dotenv import load_dotenv
from telegram.ext import MessageHandler, filters

load_dotenv()
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

set_seed(42)

# Настройка логирования: вывод в консоль и в файл
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Удаляем существующие обработчики, если есть
if logger.hasHandlers():
    logger.handlers.clear()

# Обработчик для консоли
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# Обработчик для файла
file_handler = logging.FileHandler("bot.log", mode='a')
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# Загружаем модели
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # задаём токен заполнения
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Модель успешно загружена!))")

reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)

application = Application.builder().token(TELEGRAM_TOKEN).build()

# Глобальный словарь для хранения истории диалога для каждого чата
chat_history = {}


def retrieve_keywords(user_input):
    keywords = [word for word in user_input.split() if len(word) > 3]
    return ' '.join(keywords)


def rerank_candidates(query, candidates):
    scored_candidates = []
    for idx, candidate in enumerate(candidates):
        inputs = reranker_tokenizer(query, candidate, return_tensors="pt", truncation=True, padding=True)
        outputs = reranker_model(**inputs)
        score = outputs.logits.item()
        scored_candidates.append((candidate, score, idx))
    ranked = sorted(scored_candidates, key=lambda x: x[1], reverse=True)
    return ranked


async def start_command(update: Update, context: CallbackContext):
    await update.message.reply_text('Привет! Я ваш интеллектуальный чат-бот. Напишите /help для получения инструкций.')


async def help_command(update: Update, context: CallbackContext):
    help_text = (
        "Доступные команды:\n"
        "/start - начать диалог\n"
        "/reset - сбросить историю диалога\n"
        "/help - справка\n"
        "Просто отправьте текстовое сообщение, и я отвечу на его основе."
    )
    await update.message.reply_text(help_text)


async def reset_command(update: Update, context: CallbackContext):
    chat_id = update.effective_chat.id
    if chat_id in chat_history:
        del chat_history[chat_id]
    await update.message.reply_text("История диалога сброшена.")

# Добавляем обработчик для фото
async def image_handler(update: Update, context: CallbackContext):
    if update.message.photo:
        # Берем самое качественное фото (последнее в списке)
        photo = update.message.photo[-1]
        file_id = photo.file_id
        file = await context.bot.get_file(file_id)
        logging.info(f"Получено фото с file_id: {file_id}")
        # Здесь можно сохранить файл или выполнить его обработку, например, через CV-модель
        await update.message.reply_text("Фото получено и обработано!")
    else:
        logging.error("Фото не найдено в сообщении.")

# Добавляем обработчик для стикеров
async def sticker_handler(update: Update, context: CallbackContext):
    if update.message.sticker:
        sticker = update.message.sticker
        sticker_id = sticker.file_id
        logging.info(f"Получен стикер с file_id: {sticker_id}")
        # Здесь можно сохранить стикер или обработать его как угодно
        await update.message.reply_text("Стикер получен!")
    else:
        logging.error("Стикер не найден в сообщении.")

def is_sticker(update: Update) -> bool:
    return bool(update.message and update.message.sticker)

async def text_message(update: Update, context: CallbackContext):
    if not update.message or not update.message.text:
        logging.error("Получено пустое сообщение от пользователя")
        return

    chat_id = update.effective_chat.id
    user_message = update.message.text
    logging.info(f"Chat {chat_id}: {user_message}")

    # Токенизируем сообщение с генерацией attention_mask
    inputs = tokenizer(user_message + tokenizer.eos_token, return_tensors='pt', padding=True)
    new_user_input_ids = inputs.input_ids
    new_user_attention_mask = inputs.attention_mask

    # Объединяем историю диалога с новым сообщением
    if chat_id in chat_history:
        combined_input_ids = torch.cat([chat_history[chat_id]['input_ids'], new_user_input_ids], dim=-1)
        combined_attention_mask = torch.cat([chat_history[chat_id]['attention_mask'], new_user_attention_mask], dim=-1)
    else:
        combined_input_ids = new_user_input_ids
        combined_attention_mask = new_user_attention_mask

    try:
        # Генерация нескольких кандидатов ответа
        generated_ids = model.generate(
            combined_input_ids,
            attention_mask=combined_attention_mask,
            max_length=1000,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,  # отключаем выборку
            num_beams=5,  # используем beam search
            early_stopping=True,
            num_return_sequences=5
        )

        history_length = combined_input_ids.shape[-1]
        candidate_responses = []
        for i in range(generated_ids.shape[0]):
            candidate = tokenizer.decode(generated_ids[i, history_length:], skip_special_tokens=True).strip()
            candidate_responses.append(candidate)

        refined_query = retrieve_keywords(user_message)
        if not refined_query:
            refined_query = user_message

        ranked_candidates = rerank_candidates(refined_query, candidate_responses)
        best_response, best_score, best_index = ranked_candidates[0]

        if not best_response.strip():
            best_response = "Извините, я не смог сформировать ответ. Попробуйте переформулировать запрос."

        # Обновляем историю диалога
        best_generated_ids = generated_ids[best_index].unsqueeze(0)
        best_attention_mask = torch.ones_like(best_generated_ids)
        chat_history[chat_id] = {
            'input_ids': best_generated_ids,
            'attention_mask': best_attention_mask
        }

        logging.info(f"Chat {chat_id} ответ: {best_response}")
        await context.bot.send_message(chat_id=chat_id, text=best_response)

    except Exception as e:
        logging.error(f"Ошибка при генерации текста: {e}")
        await context.bot.send_message(chat_id=chat_id, text="Произошла ошибка, попробуйте ещё раз.")


# Регистрируем обработчики команд и сообщений
start_command_handler = CommandHandler('start', start_command)
help_command_handler = CommandHandler('help', help_command)
reset_command_handler = CommandHandler('reset', reset_command)
text_message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, text_message)
image_message_handler = MessageHandler(filters.PHOTO, image_handler)
sticker_message_handler = MessageHandler(is_sticker, sticker_handler)

application.add_handler(start_command_handler)
application.add_handler(help_command_handler)
application.add_handler(reset_command_handler)
application.add_handler(text_message_handler)
application.add_handler(image_message_handler)
application.add_handler(sticker_message_handler)

application.run_polling()
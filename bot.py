import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed

from dotenv import load_dotenv
import os

load_dotenv()  # Загружает переменные из .env
# API токен Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

set_seed(42)

#Модель
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Модель успешно загружена!))")

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

application = Application.builder().token(TELEGRAM_TOKEN).build()

# Обработка команд
async def start_command(update: Update, context: CallbackContext):
    await update.message.reply_text('Привет')

async def text_message(update: Update, context: CallbackContext):
    user_message = update.message.text

    #Генерация ответа
    inputs = tokenizer(user_message, return_tensors='pt')
    outputs = model.generate(
        inputs['input_ids'],
        max_length=100,
        num_return_sequences=1,
        no_repeat_ngram_size=3,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Удаление пробелов/символов
    response = response.strip().replace('\n', ' ').replace('�', '')

    await context.bot.send_message(chat_id=update.effective_chat.id, text=response)

# Хендлеры
start_command_handler = CommandHandler('start', start_command)
text_message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, text_message)

application.add_handler(start_command_handler)
application.add_handler(text_message_handler)

# Запуск бота
application.run_polling()
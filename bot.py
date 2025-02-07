import logging
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, CallbackContext
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, set_seed, AutoModelForSequenceClassification

from dotenv import load_dotenv
import os

load_dotenv()
# API токен Telegram
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")

set_seed(42)

#Модель
model_name = "microsoft/DialoGPT-medium"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name)
print("Модель успешно загружена!))")

text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

#реранкер
reranker_model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_name)
reranker_model = AutoModelForSequenceClassification.from_pretrained(reranker_model_name)

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)

application = Application.builder().token(TELEGRAM_TOKEN).build()

def retrieve_keywords(user_input):
    keywords = [word for word in user_input.split() if len(word) > 3]
    return ' '.join(keywords)

def rerank_candidates(query, candidates):
    #ранжирование
    scores = []
    for candidate in candidates:
        inputs = reranker_tokenizer(query, candidate, return_tensors="pt", truncation=True, padding=True)
        outputs = reranker_model(**inputs)
        scores.append(outputs.logits.item())
    ranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
    return ranked

# Обработка команд
async def start_command(update: Update, context: CallbackContext):
    await update.message.reply_text('Привет')

async def text_message(update: Update, context: CallbackContext):
    user_message = update.message.text

    # Используем ретривер для извлечения ключевых слов
    refined_query = retrieve_keywords(user_message)
    if not refined_query:
        refined_query = user_message

    #генерация кандидатов
    try:
        inputs = tokenizer(refined_query, return_tensors='pt', padding=True, truncation=True)
        attention_mask = inputs['attention_mask']

        outputs = model.generate(
            inputs['input_ids'],
            attention_mask=attention_mask,
            max_length=50,
            num_beams=5,
            temperature=0.7,
            top_p=0.95,
            do_sample=True, 
            num_return_sequences=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            pad_token_id=tokenizer.eos_token_id
        )

        candidates = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        ranked_candidates = rerank_candidates(refined_query, candidates)
        best_response = ranked_candidates[0][0]

        await context.bot.send_message(chat_id=update.effective_chat.id, text=best_response)

    except Exception as e:
        logging.error(f"Ошибка при генерации текста: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Произошла ошибка, попробуйте ещё раз.")
# Хендлеры
start_command_handler = CommandHandler('start', start_command)
text_message_handler = MessageHandler(filters.TEXT & ~filters.COMMAND, text_message)

application.add_handler(start_command_handler)
application.add_handler(text_message_handler)

# Запуск бота
application.run_polling()
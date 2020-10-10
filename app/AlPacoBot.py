import telebot
import os

from PIL import Image
from decouple import config


TOKEN = config('TELEGRAM_TOKEN')
if TOKEN is None:
    print("TELEGRAM_TOKEN environment variable wasn't found, exiting")
    exit(1)
bot = telebot.TeleBot(TOKEN)

keyboard1 = telebot.types.ReplyKeyboardMarkup(True)
keyboard1.row('Привет', 'Пока', 'Работай', 'Альпака')

@bot.message_handler(commands=['start'])
def start_message(message):
    bot.send_message(message.chat.id, 'Привет! Используй клавиатуру или пришли мне свое фото!', reply_markup=keyboard1)

@bot.message_handler(content_types=['text'])
def send_text(message):
    if message.text.lower() == 'привет':
        #bot.send_message(message.chat.id, 'Привет, мой создатель')
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAMgX4AfNa_VmmHGp9bDaCiy2hp8wCQAAjEAA_AR0xQPREjTvDM0lBsE')
    elif message.text.lower() == 'пока':
        #bot.send_message(message.chat.id, 'Прощай, создатель')
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAMWX4AeLmKDUnrjMLT_f8PnF4Qacq4AAi8AA_AR0xRPXojnvS-LGxsE')
    elif message.text.lower() == 'альпака':
        photo = open('data/alpacas/9e5160657896fa20_orig.png', 'rb')
        bot.send_photo(message.chat.id, photo)
        bot.send_photo(message.chat.id, "FILEID")
    elif message.text.lower() == 'работай':
        bot.send_sticker(message.chat.id, 'CAACAgIAAxkBAAM4X4AikHyAyNOpLyGksSPBbbY5LmkAAiYAA_AR0xSFjVFXUWjSNRsE')

@bot.message_handler(content_types=['photo'])
def handle_docs_photo(message):
    try:
        file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)
        downloaded_file = bot.download_file(file_info.file_path)

        src=file_info.file_path; # сохраняет рядом со скриптом в подпапку "photos"
        with open(src, 'wb') as new_file:
           new_file.write(downloaded_file)
        bot.reply_to(message,"Фото добавлено")

    except Exception as e:
        bot.reply_to(message,e )
#'''
# можно получить в консоли id стикера
@bot.message_handler(content_types=['sticker'])
def sticker_id(message):
    print(message)
#'''

bot.polling()

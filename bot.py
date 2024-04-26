import telebot
from creds import tg_bot_token
class TelegramBot:
    def __init__(self, token):
        self.bot = telebot.TeleBot(token)
        
        @self.bot.message_handler(commands=['start'])
        def start(message):
            welcome_message = ("Привет, учёный! Мы - твоя команда умных ассистентов, "
                               "призванных упростить исследовательскую науку. "
                               "Не стесняйся задавать вопрос, коллега!")
            self.bot.send_message(message.chat.id, welcome_message)
        
        @self.bot.message_handler(func=lambda message: True)
        def handle_message(message):
            query = message.text
            response = self.process_request(query)
            self.bot.send_message(message.chat.id, response)
    
    def process_request(self, query):
        # Простая реализация абстрактного метода

# Токен вашего бота, который вы получили от BotFather
bot_token = tg_bot_token

# Создание экземпляра TelegramBot с использованием токена
bot = TelegramBot(bot_token)

# Запуск бота
bot.bot.polling()

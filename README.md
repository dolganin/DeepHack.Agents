# Решение команды "513 на базе"!

Это реализация нашего решения для хакатона DeepHack.Agents с использованием команды crew_ai.
Решение состоит из двух агентов - первый, Технический Писатель, формализует и конкретизирует запрос, обогащая его дополнительными данными.
После этого запрос уходит к Оракулу, который применяет технологию GigaChain, и в ходе решения цепи находит нужное решение.

## Установка Win11
``` bash
git clone git@github.com:dolganin/DeepHack.Agents.git
cd .\DeepHack.Agents\
python -m venv deephack
.\deephack\Scripts\activate
pip install -r requirements.txt
playwright install
playwright install-deps     
```

## Установка Unix
```bash
git clone git@github.com:dolganin/DeepHack.Agents.git
cd DeepHack.Agents\
python3 -m venv deephack
. deephack/bin/activate
pip3 install -r requirements.txt
playwright install
playwright install-deps
```


## Запуск
```python
python3 crew_ai.py
# После этого необходимо ввести ваш запрос для нашей команды
```


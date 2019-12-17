# Первое задание студтрека

Установка pipenv (это виртуальная среда pip) для debian-подобных операционных систем:

    sudo apt install pipenv
или
    pip install pipenv

Скачиваем архив  
...  
Переходим в директорию проекта  
...  
И устанавливаем зависимости (прописаны в Pipfile):

    pipenv install

Запускаем скрипт:

    pipenv shell  # Активируем виртуальное окружение
    python cybermans_PSU.py имя.jpg
    exit  # Выход из pipenv

Или так запускаем (так же из директории проекта):
    
    pipenv run python cybermans_PSU.py имя.jpg

   
## Внимание:
 1. Лабиринты изначально должны быть представимы как сетка 17на17, а не как 
 17на18(p1.jpg) или 16на17(p2.jpg)
 2. Можно установить зависимости не через pipenv, а через pip (не рекомендуется): pip3 install opencv-python==3.4.3.18
 numpy colorama

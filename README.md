# Первое задание студтрека

Установка pipenv (это виртуальная среда pip) для debian-подобных операционных систем:

    sudo apt install pipenv

Скачиваем архив  
...  
Переходим в директорию проекта  
...  
И устанавливаем зависимости (прописаны в Pipfile):

    pipenv install

Активируем виртуальное окружение:

    pipenv shell

Запускаем скрипт:

    python cybermans_PSU.py имя.jpg

Выход из pipenv:

    exit

## Недочеты:
 1. Некоторые лабиринты изначально неправильные - имеют размер сетки 
 17на18(p1.jpg) или 16на17(p2.jpg) -
 это недочет тех, кто составлял задания
 2. В Pipfile указан python3.7, потому что на моей машине не устанавливается python3.6
 (но pipenv умный и подстроит зависимости под установленный на машине python)

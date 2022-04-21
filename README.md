# intelligent_placer

## Постановка задачи

Необходимо по поданной на вход фотографии нескольких предметов на светлой горизонтальной поверхности и многоугольнику понимать, можно ли расположить одновременно все эти предметы на плоскости так, чтобы они влезли в этот многоугольник. Предметы и горизонтальная поверхность, которые могут оказаться на фотографии, заранее известны. Многоугольник задается фигурой, нарисованной темным маркером на белом листе бумаги, сфотографированной вместе с предметами.

**На вход:** фотография предметов с многоугольником в формате jpg без сжатия

**На выходе:** true или false (в зависимости от того, входят ли предметы в заданный многоугольник или нет соответсвенно) в текстовый файл answer_<name_of_the_input_image>.txt


## Требования 

### К фото в общем (фотометрические):

- Формат jpg, без сжатия
- Высота съемки: 35 - 50см
- Снимки должны быть хорошего качества - не шумное, резкое, не смазанное, объекты должны быть на 100% в фокусе
- Предметы должны быть сфотографированы сверху, допускается небольшое отклонение (не больше 5-6°)
- Фотографии должны быть сделаны с одного устройства
- Освещение: искусственное или естественное, без резких теней, на фото не должно быть пересвеченных или абсолютно черных областей

### К поверхности: 

- Горизонтальное положение
- Одна для всех фотографий
- Чистый белый лист бумаги А4 должен быть виден целиком, углы и края листа должны быть видны без перекрытий 

### По расположению объектов на фотографии:

- На фотографии не должно быть лишних предметов, кроме тех, которые были известны заранее
- Предметы не должны перекрывать друг друга (расстояние не менее 1 см между границ предметов)
- Границы предметов должны четко выделяться на фоне поверхности
- На фото предмет может присутствовать один раз
- Предметы распологаются на одном белом чистом листе бумаги А4, многоугольник нарисован на другом чистом белом листе бумаги А4

### К многоугольнику:

- Многоугольник задается фигурой, нарисованной темным маркером на белом листе бумаги А4, сфотографированной вместе с предметами
- Толщина линии маркера до 5 мм
- Число вершин многоугольника должно быть не более 10

## Датасет
https://drive.google.com/drive/folders/1pyRJifMwTl_V5w4qFpgJU2ZEaD7Etl8o?usp=sharing 


# Алгоритм

## План реализации:

### Выделение границ листов бумаги а4, многоугольника и предметов

1. Сглаживаем изображение с помощью фильтра Гаусса
2. Выделяем границы с помощью алгоритма Canny
3. Сохраняем маски выделенных границ 
4. Ищем все контуры границ с помощью встроенного метода OpenCV (findContours)
5. Оставляем только самые большие внешние контуры
6. Находим центры этих контуров и соотносим самый верхний контур с листом бумаги, на котором нарисован многоугольник, а самый нижний с листом, на котором лежат предметы
7. Внутри верхнего контура выделяем контур полигона, внутри нижнего - контуры предметов
8. Избавляемся от лишних контуров внутри контуров предметов


### Основной алгоритм размещения предметов в многоугольнике
1. Находим площадь многоугольника
2. Находим суммарную площадь предметов
3. Сравниваем полученные площади - если суммарная площадь предметов больше площади многоугольника, возвращаем False, иначе проверяем, поместятся ли предметы в многоугольник




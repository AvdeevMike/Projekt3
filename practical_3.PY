from keras.datasets import mnist
from keras import models
from keras import layers

# Загрузка данных
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Разделение данных
train_images = train_images.reshape((60000, 28*28))
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32') / 255

# Создание модели
model = models.Sequential()
model.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Обучение модели
history = model.fit(train_images, train_labels, epochs=5, batch_size=128)

# Оценка производительности модели
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Точность на тестовых данных:', test_acc)
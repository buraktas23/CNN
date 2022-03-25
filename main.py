from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D,MaxPooling2D,Activation,Dense,Dropout,Flatten
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator,img_to_array,load_img
from glob import glob
import matplotlib.pyplot as plt

# if you want to run your code on your google drive account you have to use this code
from google.colab import drive
drive.mount('/gdrive')
%cd /gdrive

train_data= "/gdrive/My Drive/Colab Notebooks/birds_CNN/train/"
test_data = "/gdrive/My Drive/Colab Notebooks/birds_CNN/test/"

img = load_img(train_data + "AMERICAN COOT/001.jpg")

img = img_to_array(img)
print(img.shape)


clas = glob(train_data + "/*")
out_num = len(clas)
print(out_num)

img_size = 0
for each in clas:
    img_size += len(glob(each + "/*"))
print(img_size)

model = Sequential()

model.add(Conv2D(32,kernel_size= (3,3),input_shape= img.shape,padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size= (3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,kernel_size= (3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(128,kernel_size= (3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(128,kernel_size= (3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,kernel_size= (3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))



model.add(Conv2D(256,kernel_size= (3,3),padding='same'))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(1024))
model.add(Activation("relu"))
model.add(Dropout(0.5))

model.add(Dense(out_num))
model.add(Activation("softmax"))


model.summary()


opt = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss="categorical_crossentropy",
              optimizer= opt,
              metrics=["accuracy"])

batch_size =128

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=45,
                                   width_shift_range=0.4,
                                   height_shift_range=0.4)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(train_data,
                                                    target_size=img.shape[:2],
                                                    batch_size=batch_size,
                                                    color_mode="rgb",
                                                    class_mode="categorical")

test_generator = test_datagen.flow_from_directory(test_data,
                                                  target_size=img.shape[:2],
                                                  color_mode="rgb",
                                                  class_mode="categorical")

hist = model.fit_generator(generator=train_generator,
                           epochs=100,
                           steps_per_epoch=514,
                           validation_steps=15,
                           validation_data=test_generator)



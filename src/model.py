from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.regularizers import l2

def build_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3,3), activation='relu',
                     input_shape=input_shape,
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu',
                     kernel_regularizer=l2(0.001)))
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu',
                    kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.4))

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
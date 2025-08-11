import tensorflow as tf
import time
import matplotlib.pyplot as plt

def load_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    
    return (x_train, y_train), (x_test, y_test)

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    return model

def train_model():
    print("Loading data...")
    (x_train, y_train), (x_test, y_test) = load_data()
    
    print("Creating model...")
    model = create_model()
    model.compile(optimizer='adam', 
                 loss='categorical_crossentropy', 
                 metrics=['accuracy'])
    
    print("Starting training...")
    start_time = time.time()
    
    history = model.fit(x_train, y_train, 
                       epochs=5, 
                       batch_size=32, 
                       validation_split=0.2)
    
    training_time = time.time() - start_time
    
    print(f"Training completed in {training_time:.2f} seconds")
    return model, history, training_time

if __name__ == "__main__":
    model, history, training_time = train_model()

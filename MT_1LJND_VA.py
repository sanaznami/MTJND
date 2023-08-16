import numpy as np
import cv2
import os
import argparse
from tensorflow import keras
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
import matplotlib.pyplot as plt

# Load and preprocess data from specified folders that has train, test and valid subfolder
def load_data(data_dir):
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    test_dir = os.path.join(data_dir, 'test')

    # Load images and labels
    def load_images_and_labels(folder_path, target_shape=(270, 480, 3)):
        images = []
        labels = []
        for subdir, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.txt'):
                    # Load label from the text file
                    label_path = os.path.join(subdir, file)
                    labels = np.loadtxt(label_path)
                    #labels.append(label_data)
                elif file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.bmp'):
                    # Load image
                    image_path = os.path.join(subdir, file)
                    image = cv2.imread(image_path)
                    # Resize the image to the target shape
                    image_resized = cv2.resize(image, (target_shape[1], target_shape[0]))
                    images.append(image_resized)
        return images, labels

    train_images, train_labels = load_images_and_labels(train_dir)
    valid_images, valid_labels = load_images_and_labels(valid_dir)
    test_images, test_labels = load_images_and_labels(test_dir)

    # Convert the lists of images and labels to numpy arrays
    train_images = np.array(train_images)
    train_labels = np.array(train_labels)
    valid_images = np.array(valid_images)
    valid_labels = np.array(valid_labels)
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return train_images, train_labels, valid_images, valid_labels, test_images, test_labels

def preprocess_data(X_Train, X_Valid, X_Test):
    #normalize data
    X_Train = np.array(X_Train, dtype='float32')
    X_Valid = np.array(X_Valid, dtype='float32')
    X_Test = np.array(X_Test, dtype='float32')
    
    X_Train /= 255
    X_Valid /= 255
    X_Test /= 255

    return X_Train, X_Valid, X_Test

# Create the model architecture
def create_model():
    # Create the model
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(270, 480, 3))

    # Freeze the layers
    for layer in base_model.layers[0:10]:
        layer.trainable = False

    MM = base_model.layers[18].output
    MM = Flatten()(MM)

    J = Dense(256, activation='relu')(MM)
    J = Dense(128, activation='relu')(J)
    J = Dense(1, activation='relu', name='JND_output')(J)

    MM_model = Model(inputs=base_model.input, outputs=[J])
    return MM_model

def train_model(MM_model, X_Train, train_labels, X_Valid, valid_labels, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs, jnd_column):
    # Compile and train the model
    optimizer = keras.optimizers.Adam(lr=learning_rate)
    MM_model.compile(optimizer=optimizer,
                    loss={'JND_output': 'mean_absolute_error'})

    checkpoint = ModelCheckpoint(filepath=checkpoint_path,
                                 monitor='val_loss',
                                 verbose=1,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='min')
    
    csv_logger = CSVLogger(csv_log_path, append=True, separator=';')

    history = MM_model.fit(X_Train,
                           {'JND_output': train_labels[:, jnd_column]},  # Use the specified column for training
                           epochs=epochs,
                           batch_size=batch_size,
                           validation_data=(X_Valid, {'JND_output': valid_labels[:, jnd_column]}),  # Use the specified column for validation
                           callbacks=[checkpoint, csv_logger],
                           shuffle=True)

def test_model(MM_model, X_Test, test_labels, result_path, model_weights_path, jnd_column):
    optimizer = keras.optimizers.Adam(lr=0.00001)
    MM_model.compile(optimizer=optimizer,
                    loss={'JND_output': 'mean_absolute_error'})
    # Test the model and save results
    MM_model.load_weights(model_weights_path)

    results = MM_model.predict(X_Test)
    print(results)

    save_test_results(results, result_path)

def save_test_results(results, result_path):
    results_array = np.array(results)
    
    np.savetxt(result_path, results_array, delimiter=',')

# Main function to execute the program    
def main(base_command, data_dir, checkpoint_path, csv_log_path, result_path, learning_rate, batch_size, epochs, model_weights_path, jnd_column):
    X_Train, train_labels, X_Valid, valid_labels, X_Test, test_labels = load_data(data_dir)
    X_Train, X_Valid, X_Test = preprocess_data(X_Train, X_Valid, X_Test)
    MM_model = create_model()

    if base_command == 'train':
        train_model(MM_model, X_Train, train_labels, X_Valid, valid_labels, checkpoint_path, csv_log_path, learning_rate, batch_size, epochs, jnd_column)
    elif base_command == 'test':
        test_model(MM_model, X_Test, test_labels, result_path, model_weights_path, jnd_column)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test the MT3LJND model.')
    parser.add_argument('base_command', choices=['train', 'test'], help='Base command for either training or testing.')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to the folder containing train, valid, and test subfolders.')
    parser.add_argument('--checkpoint_path', type=str, help='Path to save checkpoints.')
    parser.add_argument('--csv_log_path', type=str, help='Path to save CSV logs during training.')
    parser.add_argument('--result_path', type=str, help='Path to save test results.')
    parser.add_argument('--model_weights_path', type=str, help='Path to the model weights file for testing.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate for optimizer.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--epochs', type=int, default=500, help='Number of training epochs.')
    parser.add_argument('--jnd_column', type=int, default=0, choices=[0, 1, 2], help='Column index for JND values (0 for JND1, 1 for JND2, 2 for JND3)')

    args = parser.parse_args()

    # Define model weights paths based on the specified jnd_column
    if args.base_command == 'test':
        jnd_weights_paths = ['/MT_JND1_VA.h5', '/MT_JND2_VA.h5', '/MT_JND3_VA.h5']
        model_weights_path = args.model_weights_path + jnd_weights_paths[args.jnd_column]
    else:
        model_weights_path = None
    

    main(args.base_command, args.data_dir, args.checkpoint_path, args.csv_log_path, args.result_path, args.learning_rate, args.batch_size, args.epochs, model_weights_path, args.jnd_column)

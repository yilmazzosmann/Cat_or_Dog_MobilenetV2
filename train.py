from pathlib import Path
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import matplotlib.pyplot as plt
import os

# Choose GPU to train 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

#Define the train directory of cat and dog image folders
PATH = Path().absolute()
train_dir = PATH / 'train'

# Define batch and input image size
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

# Define learning rate and number of epochs
base_learning_rate = 0.0001
initial_epochs = 10
fine_tune_epochs = 10
total_epochs = initial_epochs + fine_tune_epochs


if __name__ == "__main__":

    # Define loss and metrics
    LOSS = keras.losses.BinaryCrossentropy(from_logits=False)
    METRICS = [
        keras.metrics.BinaryAccuracy(name = 'binary_accuracy'),
        keras.metrics.TruePositives(name= 'tp'),
        keras.metrics.FalsePositives(name = 'fp'),
        keras.metrics.TrueNegatives(name = 'tn'),
        keras.metrics.FalseNegatives(name = 'fn'),
        keras.metrics.Precision(name = 'precision'),
        keras.metrics.Recall(name = 'recall'),
        keras.metrics.AUC(name = 'auc'),
        keras.metrics.AUC(name = 'prc', curve='PR') #precision-recall curve
    ]

    # Define logs and model checkpoints
    def get_callbacks(phase, tensorboard_path = "./logs", ckpt_path = './checkpoint/ckpt'):
        CALLBACKS = [
            tf.keras.callbacks.TensorBoard(Path(tensorboard_path) / phase, write_graph = False),
            tf.keras.callbacks.ModelCheckpoint(filepath = ckpt_path, save_weights_only = True)
        ]
        return CALLBACKS

    ##### Data Preperation #####
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                shuffle = True,
                                                                seed = 42,
                                                                batch_size = BATCH_SIZE,
                                                                image_size = IMG_SIZE)
    class_names = train_ds.class_names
    
    # Chose %20 for the validation
    val_batches = tf.data.experimental.cardinality(train_ds)
    validation_ds = train_ds.take(val_batches // 5)
    train_ds = train_ds.skip(val_batches // 5)

    # Visualize training images
    plt.figure(figsize=(10,10))
    for (images, labels) in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    plt.savefig("1_training_imgs.png", dpi=300)

    print("Number of training batches: %d" % tf.data.experimental.cardinality(train_ds))
    print("Number of validation batches: %d" % tf.data.experimental.cardinality(validation_ds))

    # Create flexible and efficient training and validation pipelines
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size = AUTOTUNE)
    validation_ds = validation_ds.prefetch(buffer_size = AUTOTUNE)
    
    # Visualize data augmentation 
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.2),
    ])

    for images, labels in train_ds.take(1):
        plt.figure(figsize=(10,10))
        first_image = images[0]
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            augmented_image = data_augmentation(
                tf.expand_dims(first_image, 0), training= True
            )
            plt.imshow(augmented_image[0].numpy().astype("int32"))
            plt.title(class_names[labels[0]])
            plt.axis("off")
    plt.savefig("2_training_imgs_augmented.png", dpi=300)

    ##### Build a NN model #####
    # Prepare MobileNetV2 base model
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                                alpha=1.0,
                                                include_top = False,
                                                weights= "imagenet")
    # Freeze the base model
    base_model.trainable = False

    # Create a new model on top
    inputs = keras.Input(shape = IMG_SHAPE)

    # Apply data augmentation
    x = data_augmentation(inputs)

    #Pretrained Xception weights require input to be scaled between [-1, 1]
    scale_layer = keras.layers.experimental.preprocessing.Rescaling(scale = 1 / 127.5, offset=-1)
    x = scale_layer(x)

    # Freeze the base model for transfer-learning and train the added head only
    x = base_model(x, training = False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.summary()

    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate= base_learning_rate),
        loss = LOSS,
        metrics = METRICS
    )

    ##### Start transfer learning and visualize the loss and accuracy #####
    history = model.fit(train_ds,
                        epochs=initial_epochs,
                        validation_data= validation_ds,
                        callbacks= get_callbacks("transfer_learning"))
    
    acc = history.history["binary_accuracy"]
    val_acc = history.history["val_binary_accuracy"]

    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc = "lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.legend(loc = "upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation loss")
    plt.xlabel("epoch")
    plt.savefig("3_res_transfer_learning.png", dpi=300)




    ##### Finetuning Stage #####
    # Fine-tune from the 90th layer onwards and freeze the before
    base_model.trainable = True
    fine_tune_at = 90
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    model.summary()

    # Compile with a lower learning rate for fine-tuning
    model.compile(
        optimizer = keras.optimizers.RMSprop(learning_rate= base_learning_rate / 10),
        loss = LOSS,
        metrics = METRICS
    )

    # Start fine-tuning training and visualize the loss and accuracy
    history_fine_tuned = model.fit(train_ds,
                        epochs = total_epochs,
                        initial_epoch= history.epoch[-1],
                        validation_data = validation_ds,
                        callbacks= get_callbacks("finetuning")
    )

    acc += history_fine_tuned.history["binary_accuracy"]
    val_acc += history_fine_tuned.history["val_binary_accuracy"]

    loss += history_fine_tuned.history["loss"]
    val_loss += history_fine_tuned.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.plot([initial_epochs-1, initial_epochs-1],
            plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc = "lower right")
    plt.ylabel("Accuracy")
    plt.ylim([0.8, 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training loss")
    plt.plot(val_loss, label="Validation loss")
    plt.plot([initial_epochs-1, initial_epochs-1],
        plt.ylim(), label="Start Fine Tuning")
    plt.legend(loc = "upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation loss")
    plt.xlabel("epoch")
    plt.savefig("4_res_fine_tuning.png", dpi=300)

    ##### Save the model and weights #####
    # This is just back-up, model weights and indexes are already saved also in checkpoint
    if not Path("./saved_model").exists():
        Path("./saved_model").mkdir()

    model.save("saved_model/cat_dog")
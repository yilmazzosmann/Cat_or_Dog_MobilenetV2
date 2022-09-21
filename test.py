from pathlib import Path
import tensorflow as tf
import pandas as pd
import os

# Choose GPU to train 
PATH = Path().absolute()
test_dir = PATH / 'test1'

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
IMG_SHAPE = IMG_SIZE + (3,)

##### Test Data Preperation #####
test_ds = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                shuffle = False,
                                                                batch_size = BATCH_SIZE,
                                                                image_size = IMG_SIZE)

# Load the model that is saved from training
model = tf.keras.models.load_model('saved_model/cat_dog')
model.summary()

# Create flexible and efficient test data pipeline
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.prefetch(buffer_size = AUTOTUNE)

##### Start Prediction #####
predictions = model.predict(test_ds)

# Create a dataframe to save corresponding file names and predictions
filenames = os.listdir("./test1/test")
test_df = pd.DataFrame({'filename' : filenames}) 
test_df['category'] = tf.where(predictions < 0.5, 0, 1)

# Save the submission as a csv file
submission_df = test_df.copy()
submission_df['id'] = submission_df['filename'].str.split('.').str[0]
submission_df['label'] = submission_df['category']
submission_df.drop(['filename', 'category'], axis=1, inplace=True)
submission_df.to_csv('submission.csv', index=False)
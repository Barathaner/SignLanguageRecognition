from keras.optimizers import RMSprop

from preprocessing.dataretrieval import *
from preprocessing.featureextraction import *
#
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from os.path import exists

# training main --> MODEL &&& testing vom CNN -- exportiere Model
# irgendwoanders visualize main (Model,webcam)--> lädt model und wertet aus und visualisiert
if __name__ == "__main__":
    # clean data and put it into /data/train/
    # dr = Dataretrieval(CleanCHALearn())
    # dr.retrieve(["sister", "friend"])
    # fe = FeatureExtraction(Skelleting_as_image())
    # fe.extractFeature("data/train")

    drtest = Dataretrieval(CleanCHALearn())
    drtest.retrieve(["sister", "friend"],destination_folder='data/test/',source_folder='data/raw/test/',source_CSV_path="data/testdata/ground_truth.csv", dest_CSV_path="data/features/skeletons/test.csv")
    fetest = FeatureExtraction(Skelleting_as_image())
    fetest.extractFeature("data/test/", "data/features/skeletons_test/")

    directory = 'data/features/skeletons_test/'
    df = pd.read_csv("data/features/skeletons/test.csv")

    # filter for images that exist
    for index, file_name in enumerate(df['file_name']):
        if (not exists(directory + file_name)):
            df = df.drop(index=index, axis=0)

    df = df.reset_index(drop=True)

    file_paths = df['file_name'].values
    labels = df['word'].values




    ds_train = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    ds_test = tf.data.Dataset.from_tensor_slices((file_paths, labels))

    def read_image(image_file, label):
        image = tf.io.read_file(directory + image_file)
        image = tf.image.decode_image(image, channels=3, dtype=tf.float32)
        return image, label


    ds_train = ds_train.map(read_image).batch(2)


    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 200x200 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(512, 512, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # # The fifth convolution
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 512 neuron hidden layer
        tf.keras.layers.Dense(512, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('dandelions') and 1 for the other ('grass')
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics='accuracy')

    history = model.fit(ds_train,
                        steps_per_epoch=8,
                        epochs=15,
                        verbose=1,
                        validation_data = ds_train,
                        validation_steps=8)

    model.evaluate(ds_train)





    #
    #
    #
    # for epoch in range(2):
    #     for x, y in ds_train:
    #         # train here
    #         pass
    #
    # model = keras.Sequential(
    #     [
    #         # Note the input shape is the desired size of the image 512x512 with 3 bytes color
    #         layers.Input((512, 512, 3)),
    #         layers.Conv2D(16, 3, padding="same"),
    #         layers.Conv2D(32, 3, padding="same"),
    #         layers.MaxPooling2D(),
    #         layers.Flatten(),
    #         layers.Dense(16),
    #     ]
    # )
    #
    # model.compile(
    #     optimizer=keras.optimizers.Adam(),
    #     loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
    #     metrics=["accuracy"],
    # )
    #
    # model.fit(ds_train, epochs=2, verbose=2)

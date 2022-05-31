from preprocessing.dataretrieval import *
from preprocessing.featureextraction import *


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
#training main --> MODEL &&& testing vom CNN -- exportiere Model
#irgendwoanders visualize main (Model,webcam)--> lädt model und wertet aus und visualisiert
if __name__ == "__main__":
    #clean data and put it into /data/train/
    #dr = Dataretrieval(CleanCHALearn())
    #dr.retrieve(["sister", "friend"])
    #fe = FeatureExtraction(Skelleting_as_image())
    #fe.extractFeature("data/train")
    directory = 'data/features/skeletons/'
    df = pd.read_csv(directory + 'train.csv')
    file_paths = df['file_name'].values
    labels = df['word'].values
    ds_train = tf.data.Dataset.from_tensor_slices((file_paths,labels))

    def read_image(image_file,label):
        image = tf.io.read_file(directory + image_file)
        image = tf.image.decode_image(image,channels=3,dtype=tf.float32)
        return image,label

    ds_train = ds_train.map(read_image).batch(2)

    for epoch in range(10):
        for x,y in ds_train:
            #train here
            pass

    model = keras.Sequential(
        [
            layers.Input((28, 28, 1)),
            layers.Conv2D(16, 3, padding="same"),
            layers.Conv2D(32, 3, padding="same"),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(10),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss=[keras.losses.SparseCategoricalCrossentropy(from_logits=True), ],
        metrics=["accuracy"],
    )

    model.fit(ds_train, epochs=10, verbose=2)



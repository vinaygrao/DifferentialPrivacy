"""
Example membership inference attack against a deep net classifier on the CIFAR10 dataset
"""
import numpy as np

from absl import app
from absl import flags

import tensorflow as tf
from tensorflow.keras import layers

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from  mial.estimators import  ShadowModelBundle, AttackModelBundle, prepare_attack_data, CustomModelBundle, ShadowModelBundle2
import mnist_dpsgd_tutorial_keras

NUM_CLASSES = 10
WIDTH = 32
HEIGHT = 32
CHANNELS = 3
SHADOW_DATASET_SIZE = 30000
ATTACK_TEST_DATASET_SIZE = 60000


FLAGS = flags.FLAGS
flags.DEFINE_integer(
    "target_epochs", 30, "Number of epochs to train target and shadow models."
)
flags.DEFINE_integer("attack_epochs", 10, "Number of epochs to train attack models.")

flags.DEFINE_integer("num_shadows", 1, "Number of epochs to train attack models.")


def get_data():
    """Prepare CIFAR10 data."""
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
    y_train = tf.keras.utils.to_categorical(y_train)
    y_test = tf.keras.utils.to_categorical(y_test)
    X_train = X_train.astype("float32")
    X_test = X_test.astype("float32")
    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")
    X_train /= 255
    X_test /= 255
    return (X_train, y_train), (X_test, y_test)
def load_mnist():
  """Loads MNIST and preprocesses to combine training and validation data."""
  (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
  X_train = np.array(X_train, dtype=np.float32) / 255
  X_test = np.array(X_test, dtype=np.float32) / 255

  X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
  X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

  y_train = np.array(y_train, dtype=np.int32)
  y_test = np.array(y_test, dtype=np.int32)

  y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
  y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

  assert X_train.min() == 0.
  assert X_train.max() == 1.
  assert X_test.min() == 0.
  assert X_test.max() == 1.

  #train_data = train_data[:FLAGS.data_slice,:]
  #train_labels = train_labels[:FLAGS.data_slice, :]
  #test_data = test_data[:FLAGS.data_slice, :]
  #test_labels = test_labels[:FLAGS.data_slice, :]

  return (X_train, y_train), (X_test, y_test)


def target_model_fn():
    """The architecture of the target (victim) model.

    The attack is white-box, hence the attacker is assumed to know this architecture too."""

    model = tf.keras.models.Sequential()

    model.add(
        layers.Conv2D(
            32,
            (3, 3),
            activation="relu",
            padding="same",
            input_shape=(WIDTH, HEIGHT, CHANNELS),
        )
    )
    model.add(layers.Conv2D(32, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())

    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(NUM_CLASSES, activation="softmax"))
    model.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model


def attack_model_fn():
    """Attack model that takes target model predictions and predicts membership.

    Following the original paper, this attack model is specific to the class of the input.
    AttachModelBundle creates multiple instances of this model for each class.
    """
    model = tf.keras.models.Sequential()
    model.add(layers.Dense(64, activation="relu", input_shape=(NUM_CLASSES,)))
    model.add(layers.Dense(128, activation="relu"))

    #model.add(layers.Dropout(0.3, noise_shape=None, seed=None))

    #model.add(layers.Dropout(0.2, noise_shape=None, seed=None))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(64, activation="relu"))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


def demo(argv):
    del argv  # Unused.

    (X_train, y_train), (X_test, y_test) = load_mnist()
#    x = np.concatenate((X_train, X_test))
 #   y = np.concatenate((y_train, y_test))

 #   train_size = 0.5
 #   X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=train_size)
    # Train the target model.
    print("Training the target model...")
    target_model = mnist_dpsgd_tutorial_keras.main()
    target_model.fit(
       X_train, y_train, epochs=FLAGS.target_epochs, validation_split=0.1, verbose=True, validation_data=(X_test, y_test),  batch_size=250
    )

    smb = ShadowModelBundle2(
        target_model,
        shadow_dataset_size=SHADOW_DATASET_SIZE,
        num_models=FLAGS.num_shadows,
    )

    # We assume that attacker's data were not seen in target's training.
    attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
        X_test, y_test, test_size=0.1
    )
    print(attacker_X_train.shape, attacker_X_test.shape)

    print("Training the shadow models...")
    X_shadow, y_shadow = smb.fit_transform(
        attacker_X_train,
        attacker_y_train,
        fit_kwargs=dict(
            epochs=FLAGS.target_epochs,
            verbose=True,
            validation_data=(attacker_X_test, attacker_y_test),
            batch_size=250
        ),
    )







    # Train the shadow models.
 #   smb = CustomModelBundle()
    # We assume that attacker's data were not seen in target's training.
 #   attacker_X_train, attacker_X_test, attacker_y_train, attacker_y_test = train_test_split(
 #       X_test, y_test, test_size=0.1
 #   )
 #   print(attacker_X_train.shape, attacker_X_test.shape)

 #   print("Training the shadow models...")
 #   X_shadow, y_shadow = smb.fit_transform(target_model,X_train, y_train)


    # ShadowModelBundle returns data in the format suitable for the AttackModelBundle.
    amb = AttackModelBundle(attack_model_fn, num_classes=NUM_CLASSES)

    # Fit the attack models.
    print("Training the attack models...")
    amb.fit(
        X_shadow, y_shadow, fit_kwargs=dict(epochs=FLAGS.attack_epochs, verbose=True)
    )

    # Test the success of the attack.

    # Prepare examples that were in the training, and out of the training.
    data_in = X_train[:ATTACK_TEST_DATASET_SIZE], y_train[:ATTACK_TEST_DATASET_SIZE]
    data_out = X_test[:ATTACK_TEST_DATASET_SIZE], y_test[:ATTACK_TEST_DATASET_SIZE]

    # Compile them into the expected format for the AttackModelBundle.
    attack_test_data, real_membership_labels = prepare_attack_data(
        target_model, data_in, data_out
    )

    # Compute the attack accuracy.
    attack_guesses = amb.predict(attack_test_data)
    attack_accuracy = np.mean(attack_guesses == real_membership_labels)

    print(attack_accuracy)
    fscores = precision_recall_fscore_support(real_membership_labels, attack_guesses)

    [print(fscore) for fscore in fscores]
    from sklearn.metrics import precision_score, recall_score, f1_score

    print(f1_score(real_membership_labels, attack_guesses))


if __name__ == "__main__":
    app.run(demo)

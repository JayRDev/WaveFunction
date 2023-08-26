import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

# Controlled Training Variables
EPOCHS = 400
BATCH_SIZE = 32
SEQUENCE_LENGTHS = [8,  4]
NUM_SEGMENTS = 5  # Number of segments in x-axis

# Define wave functions and their corresponding labels
wave_functions = [np.sin, np.cos, lambda x: np.clip(np.tan(x), -5, 5), np.exp, np.log1p]
wave_labels = [np.eye(len(wave_functions))[i] for i in range(len(wave_functions))]

# Generate combined data
xpts = np.linspace(0, 10, num=500).reshape(-1, 1)
segment_length = len(xpts) // NUM_SEGMENTS
ypts_combined = []
wave_types_combined = []
for i in range(NUM_SEGMENTS):
    func_idx = np.random.randint(len(wave_functions))
    y_segment = wave_functions[func_idx](xpts[i * segment_length: (i + 1) * segment_length])
    y_segment = np.clip(y_segment, -100, 100)  # Clip the values within the range
    ypts_combined.append(y_segment)
    wave_types_combined.extend([wave_labels[func_idx]] * segment_length)

ypts_combined = np.concatenate(ypts_combined)
wave_types_combined = np.array(wave_types_combined)

def custom_loss(y_true, y_pred):
    mse = tf.keras.losses.mean_squared_error(y_true, y_pred)
    mae = tf.keras.losses.mean_absolute_error(y_true, y_pred)
    return 0.5 * mse + 0.5 * mae

def reshape_data(xpts, ypts, sequence_length, wave_types):
    x = np.array([xpts[i: i + sequence_length] for i in range(len(xpts) - sequence_length + 1)])
    y_classification = wave_types[:len(x)]
    y_regression = ypts[:len(x)]
    return x, {'classification_output': y_classification, 'regression_output': y_regression}

# Define a model that can handle varying sequence lengths
def flexible_model_fn():
    inputs = Input(shape=(None, 1))
    x = tf.keras.layers.LSTM(50, return_sequences=True)(inputs)
    x = tf.keras.layers.LSTM(50)(x)
    x = tf.keras.layers.Dense(200, activation='relu')(x)
    classification_output = tf.keras.layers.Dense(len(wave_functions), activation='softmax', name='classification_output')(x)
    regression_output = tf.keras.layers.Dense(1, name='regression_output')(x)

    model = Model(inputs=inputs, outputs=[classification_output, regression_output])
    return model

# Create the model instance
model = flexible_model_fn()

# Compile the model
model.compile(optimizer='adam',
              loss={'classification_output': 'categorical_crossentropy', 'regression_output': custom_loss},
              loss_weights={'classification_output': 1., 'regression_output': 1.})

# Training and Visualization
for seq_len in SEQUENCE_LENGTHS:
    print(f"Training with sequence length {seq_len}")

    # Create a plot outside the epoch loop
    fig, ax = plt.subplots(figsize=(8, 6))
    line1, = ax.plot([], [], label='Original')
    line2, = ax.plot([], [], label='Fitted')
    ax.legend()
    plt.ion()  # Turn on interactive mode
    plt.show()

    for epoch in range(EPOCHS):
       # Update the original wave every 10 epochs
        if epoch % 10 == 0:
            ypts_combined = []
            for i in range(NUM_SEGMENTS):
                func_idx = np.random.randint(len(wave_functions))
                y_segment = wave_functions[func_idx](xpts[i * segment_length: (i + 1) * segment_length])
                y_segment = np.clip(y_segment, -100, 100)  # Clip the values within the range
                ypts_combined.append(y_segment)
            ypts_combined = np.concatenate(ypts_combined)
            x_train, y_train = reshape_data(xpts, ypts_combined, seq_len, wave_types_combined)

        model.fit(x_train, y_train, epochs=1, batch_size=BATCH_SIZE, verbose=0)
        classification_predictions, regression_predictions = model.predict(x_train)

        # Update the plot
        y_min = min(ypts_combined.min(), regression_predictions.min()) - 1
        y_max = max(ypts_combined.max(), regression_predictions.max()) + 1
        ax.set_ylim([y_min, y_max])
        line1.set_xdata(xpts[:len(regression_predictions)])
        line1.set_ydata(ypts_combined[:len(regression_predictions)])
        line2.set_xdata(xpts[:len(regression_predictions)])
        line2.set_ydata(regression_predictions)
        ax.set_title(f"Sequence Length: {seq_len} | Epoch: {epoch + 1}")
        ax.relim()
        ax.autoscale_view()
        plt.draw()
        plt.pause(0.1)  # Pause to allow time to view the plot

plt.ioff()  # Turn off interactive mode

# Save the model
save_model = input("Do you want to save the model? (yes/no): ")
if save_model.lower() == 'yes':
    model_path = "wave_model"
    model.save(model_path)
    print(f"Model saved to {model_path}")
else:
    print("Model not saved.")
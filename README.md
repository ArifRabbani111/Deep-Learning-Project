MNIST CNN Model with Hyperparameter Tuning
This notebook demonstrates the process of building, training, and tuning a Convolutional Neural Network (CNN) model for classifying handwritten digits from the MNIST dataset.

Steps Performed:
Load the MNIST Dataset: The MNIST dataset was loaded using TensorFlow's built-in datasets.
Preprocess the Data: The image data was normalized and reshaped to be suitable for the CNN model.
Build a CNN Model: A basic CNN architecture was defined using tensorflow.keras.Sequential with Conv2D, MaxPooling2D, Flatten, and Dense layers.
Compile the Model: The model was compiled with the Adam optimizer, sparse categorical crossentropy loss, and accuracy as the metric.
Train the Model: The initial CNN model was trained on the training data and validated on the test data.
Visualize Results and Insights: Predictions on a few test images were displayed to visually assess the model's performance.
Evaluate the Model: The performance of the initial model was evaluated on the test dataset.
Hyperparameter Tuning:
A hyperparameter search space was defined using Keras Tuner's RandomSearch.
The CNN model architecture was modified within a function (build_cnn_model) to accept hyperparameters as arguments, including additional layers (Batch Normalization, another Conv2D layer) and Dropout for regularization.
The hyperparameter tuning process was implemented using RandomSearch to find the best combination of hyperparameters based on validation accuracy.
The best hyperparameters found by the tuner were retrieved.
Build and Train the Model with Best Hyperparameters: A new model was built using the best hyperparameters and trained on the complete training dataset.
Evaluate the Best Model: The performance of the final model with the best hyperparameters was evaluated on the test dataset.
Results:
The notebook shows the training progress and the final evaluation metrics (loss and accuracy) for both the initial model and the model trained with the best hyperparameters. The best hyperparameters found during the tuning process are also displayed

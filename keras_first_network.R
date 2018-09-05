library(keras)
source("https://bit.ly/2wHJ3AU")

mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
validation_images <- mnist$test$x
validation_labels <- mnist$test$y

# see what's going on with the input encoding
array_print(train_images)

# more rows and cols shown
array_print(train_images, num_2d_rows = 28, num_2d_cols = 28)

# plot the first entry, letting the function know the largest value it needs to plot is 255
plot(as.raster(train_images[1, , ], max = 255))

# what's going on with the labels? They're just numbers to match the drawings
array_print(train_labels)

## Same for the test data, just 10000 samples rather than 60000:
array_print(validation_images)
array_print(validation_labels)

## Reformat image data to flatten each sample and scale values to 0 to 1
train_images <- array_reshape(train_images, dim = c(60000, 28 * 28))  # 28 * 28 = 784
train_images <- train_images / 255

validation_images <- array_reshape(validation_images, dim = c(10000, 28 * 28))  # 28 * 28 = 784
validation_images <- validation_images / 255

## Now see the formatting for train_images
## since we flattened, we'll tell the function it shouldn't try to use any 2d representation
array_print(train_images, use_2d = FALSE, n = 6)


## Now we can encode the labels as "one-hot" vectors
## just to remember what these look like:
array_print(train_labels, n = 6)

train_labels <- to_categorical(train_labels)
array_print(train_labels, use_2d = FALSE, n = 10)

## And same for test labels:
validation_labels <- to_categorical(validation_labels)

## Build the network!!
network <- keras_model_sequential() %>%
  layer_dense(units = 512, activation = "relu", input_shape = c(28 * 28)) %>%   # 28 * 28 = 784
  layer_dense(units = 10, activation = "softmax")

## Compile the network, defining optimizer method and loss function
network %>% compile(
  optimizer = optimizer_rmsprop(lr = 0.001, rho = 0.9),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

validation_data_list <- list(validation_images, validation_labels)

## Train the network, tracking accuracy on both the training data and the testing data
network %>% fit(train_images, train_labels, epochs = 5, batch_size = 64, validation_data = validation_data_list)

## Evaluate
metrics <- network %>% evaluate(validation_images, validation_labels)
print(metrics)

## compare predictions to actual labels:
predictions <- network %>% predict_classes(validation_images[1:10, ])
array_print(predictions, n = 10, use_2d = FALSE)
array_print(validation_labels, n = 10, use_2d = FALSE)



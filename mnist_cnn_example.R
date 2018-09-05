library(keras)
source("https://bit.ly/2wHJ3AU")

mnist <- dataset_mnist()
train_images <- mnist$train$x
train_labels <- mnist$train$y
test_images <- mnist$test$x
test_labels <- mnist$test$y

# we don't want to serialize the input, we want to leave them as 2d images. But we still need to scale the values.
train_images <- train_images / 255
test_images <- test_images / 255

# the labels are still one-hot vectors
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

#### 
#### We *do* have to do some reshaping of the input still. Currently our images are rank-3 tensors,
#### (samples, rows, cols)
array_print(train_images)

#### But, convolutional neural nets expect all inputs to have channels, ie, be a rank-4 tensor,
#### (samples, rows, cols, channels). Our mnist data is in black and white, so it only has one channel, but we
#### still need to create it as a new dimension.
train_images <- array_reshape(train_images, dim = c(60000, 28, 28, 1))
test_images <- array_reshape(test_images, dim = c(10000, 28, 28, 1))

#### we won't try to see it with array_print(), because it won't be very helpful. But remember how we can
#### think of tensors, a tensor of rank N is a collection of tensors of rank N-1. 
#### So train_images is a collection of 60000 tensors, each of which is a 28x28x1 tensor. 


## Build the network!!
network <- keras_model_sequential() %>%
  # first convolutional layer: 32 filter channels output based on 3x3 window
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>%
  # first max pooling layer
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  # second convolution and max_pool layers... more channels
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  
  # flatten to a 1d tensor
  layer_flatten() %>%
  # a dense hidden layer for learning from the convolutional output
  layer_dense(units = 64, activation = "relu") %>%
  # output layer, softmax to get a vector of probabilities
  layer_dense(units = 10, activation = "softmax")
  
## Compile the network, defining optimizer method and loss function
network %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

test_data_list <- list(test_images, test_labels)

## Train the network, tracking accuracy on both the training data and the testing data
network %>% fit(train_images, train_labels, epochs = 5, batch_size = 64, validation_data = test_data_list)

## Evaluate
metrics <- network %>% evaluate(test_images, test_labels)
print(metrics)

## compare predictions to actual labels for the first 10 images:
predictions <- network %>% predict_classes(test_images[1:10, 1:28, 1:28, 1, drop = FALSE]) # if a dimenion only has one entry (as is the case with our (60000, 28, 28, 1) tensor), R will silently drop it during subsetting. Thanks R. To turn this off, add a , drop = FALSE. 
array_print(predictions, n = 10, use_2d = FALSE)
array_print(test_labels, n = 10, use_2d = FALSE)



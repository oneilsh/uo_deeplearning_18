# provides a few handy functions, including array_print(), fasta_to_tensor(), array_c()
source("tensorfuncs.R")
library(keras)

## First, read the fasta files in, and see what we've got
random_dna_tensor <- fasta_to_tensor("random_dna_seqs.fasta")
array_print(random_dna_tensor)
promoter_tensor <- fasta_to_tensor("promoter_seqs.fasta")
array_print(promoter_tensor)

##########################
## We gotta make some training & validation data. Feel free to skip this section, it's just annoying data wrangling.
## First let's concatenate the two 5000-sample tensors to make one 10000-sample tensors
both_tensor <- array_c(promoter_tensor, random_dna_tensor)

## Now let's create a vector of 5000 1s and 5000 0s (to indicate the first 5000 samples are promoter sequences,
## And the second 5000 are random dna sequences)
labels <- array_reshape(c(rep(1, 5000), rep(0, 5000)), dim = c(10000)) # complicated for c(rep(1, 5000), rep(0, 5000))

## Now let's shuffle both the tensor and the labels, but use the same random order for each
random_order <- sample(1:10000)   # shuffled vector of integers
shuffled_both_tensor <- both_tensor[random_order, ,]
shuffled_labels <- labels[random_order]

## Now we can get training tensors and lables, and validation tensors and labels
train_x <- array_reshape(shuffled_both_tensor[1:8000,,], dim = c(8000, 51, 4))
train_y <- array_reshape(shuffled_labels[1:8000], dim = c(8000))
validation_x <- array_reshape(shuffled_both_tensor[8001:10000,,], dim = c(2000, 51, 4))
validation_y <- array_reshape(shuffled_labels[8001:10000], dim = c(2000))

array_print(train_x, n = 4)
## End generation of training and validation data section.
######################################


# Convolutional layers treat each entry as a "pixel" with multiple channels; here we're treating each base as a "pixel" with
# four channels. We're also adding regularization to the some of the weights: this adds loss corresponding to the size
# of the weights in those layers, which drives the model to reduce as many weights as it can to learn from, this can
# help to keep models from "memorizing" the training data.
# 
# the "dropout" layer serves the same purpose, by intentionally setting random weights to 0 (temporarily) during training,
# which forces the model to not rely too much on any given weight in the network. This can help the network generalize.
model <- keras_model_sequential() %>%
  layer_conv_1d(filters = 16, kernel_size = 3, activation = "relu", input_shape = c(51, 4), kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_dropout(rate = 0.5) %>%
  layer_conv_1d(filters = 32, kernel_size = 3, activation = "relu", input_shape = c(51, 4), kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_max_pooling_1d(pool_size = 2) %>%
  layer_flatten() %>%
  layer_dense(units = 64, activation = "relu", kernel_regularizer = regularizer_l2(0.001)) %>%
  #layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dense(units = 1, activation = "sigmoid")


  
## Basic optimizer, loss function appropriate for probility predictions of binary classes
model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

## Train! Save the losses and accuracies for later looking at
history <- model %>% fit(train_x, train_y, 
                         epochs = 10, 
                         batch_size = 128, 
                         validation_data = list(validation_x, 
                                                validation_y))

head(model %>% predict(test_x), n = 20)
head(test_y, n = 20)


# provides a few handy functions, including array_print(), fasta_to_tensor(), array_c()
source("tensorfuncs.R")
library(keras)

## First, read the fasta files in, and see what we've got
random_dna_tensor <- fasta_to_tensor("random_dna_seqs.fasta")
array_print(random_dna_tensor)
promoter_tensor <- fasta_to_tensor("promoter_seqs.fasta")
array_print(promoter_tensor)

## We gotta make some training data. First let's concatenate the two 5000-sample tensors to make one 10000-sample tensors
both_tensor <- array_c(promoter_tensor, random_dna_tensor)

## Now let's create a vector of 5000 1s and 5000 0s (to indicate the first 5000 samples are promoter sequences,
## And the second 5000 are random dna sequences)
labels <- array_reshape(c(rep(1, 5000), rep(0, 5000)), dim = c(10000)) # complicated for c(rep(1, 5000), rep(0, 5000))

## Now let's shuffle both the tensor and the labels, but use the same random order for each
random_order <- sample(1:10000)   # shuffled vector of integers
shuffled_both_tensor <- both_tensor[random_order,,]
shuffled_labels <- labels[random_order]

## Now we can get training tensors and lables, and validation tensors and labels
train_x <- array_reshape(shuffled_both_tensor[1:8000,,], dim = c(8000, 51, 4))
train_y <- array_reshape(shuffled_labels[1:8000], dim = c(8000))
validation_x <- array_reshape(shuffled_both_tensor[8001:10000,,], dim = c(2000, 51, 4))
validation_y <- array_reshape(shuffled_labels[8001:10000], dim = c(2000))

## This LSTM model is simple at one layer, but even simple LSTM layers are complex on the inside!
## The LSTM layer takes the shape of a single input entry of a sequence, in this case the shape of a single
## dna base, which is a one-hot vector of length 4; the LSTM process each base one at a time, "remembering" 
## what it previously output for each base, for use in determine output for the next base.
## Because of this internal looping, they tend to be very computationally intensive to train.
##
## It's a complex thing, but it tends to work well when the input has "time structure" or "order structure"
## -- think sentences, frames of videos, or in this case, DNA-bases.

## We want to predict a binary output (label = 1 or label = 0), 
## so the last layer is a sigmoid output to turn the result into a probability-like value
model <- keras_model_sequential() %>%
 layer_lstm(units = 4) %>%
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

head(model %>% predict(validation_x), n = 20)
head(validation_y, n = 20)



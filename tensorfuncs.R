library(abind)
library(lgcp)
library(seqinr)
library(stringr)


### This "package", such as it is, is a collection of handy functions for working with tensors and keras
### Most of the functions are hastily written hacks in this version, but I intend to build them into a larger,
### more robust suite of helper functions.
### Shawn T. O'Neil, Oregon State University, Sep 5 2018

# PS - ignore the "named" tensor stuff, I'm still working out the best way to do that.


# given the name of a fasta file (e.g. "some_fasta.fa"), returns a tensor of shape
# [num_sequences, seqs_length, 4], representing 1-hot encoding of the data
# requires seqinr and stringr packages
# (doing some nasty R tricks for speed, and this could use a rewrite+comments)
fasta_to_tensor <- function(fasta_file) {
  dna_seqs <- read.fasta(file = fasta_file, as.string = TRUE, forceDNAtolower = FALSE)
  dna_vec <- unlist(dna_seqs)
  num_seqs <- length(dna_seqs)
  if(length(unique(unlist(lapply(dna_vec, nchar)))) > 1) {
    stop("Sorry, all of the sequences in the fasta file must be the same length to turn them into a tensor.")
  }
  seqs_length <- nchar(dna_seqs[1])
  dna_split_list <- str_split(dna_vec, "")
  dna_matrix <- do.call(rbind, dna_split_list)
  dna_spread <- array_reshape(dna_matrix, dim = prod(dim(dna_matrix)))
  one_hot <- lapply(dna_spread, 
                    function(letter) {
                      if(letter == "A") {
                        return(c(1, 0, 0, 0))
                      } else if(letter == "C") {
                        return(c(0, 1, 0, 0))
                      } else if(letter == "G") {
                        return(c(0, 0, 1, 0))
                      } else {
                        return(c(0, 0, 0, 1))
                      }
                    })
  flattened <- unlist(one_hot)
  dna_tensor <- array_reshape(flattened, dim = c(num_seqs, seqs_length, 4))
  return(dna_tensor)
}

# Allows a user to "name" the dimenions in a tensor; not quite the same as R dimnames
# though, because that would be e.g. matrix(0, nrow = 3, ncol = 2, dimnames = list(c("y", "y", "y"), c("x", "x")))
# requiring repeating the "dimension" name over each entry. Here, by contrast:
# tensor <- array_reshape(data, dim = c(100, 60, 400, 300))
# tensor <- array_name_dims(tensor, c("samples", "frames", "y", "x"))
array_name_dims <- function(tensor, names) {
  if(any(length(names) != length(dim(tensor)))) { 
    stop(paste0("Mismatched rank. Dimensions of tensor: ", 
                paste0(dim(tensor), collapse = ", "), 
                "; attempted names: ", 
                paste0(names, collapse = ", "),
                "\n\n"
    )
    )
  }
  for(index in 1:length(names)) {
    dimnames(tensor)[[index]] <- rep(names[index], dim(tensor)[index])
  }
  return(tensor)
}

# Whew, brutal. This is a recursive function for printing tensors (as used by tensorflow) sanely in R. 
array_print <- function(tensor, n = 3, digits = 3, num_2d_rows = 6, num_2d_cols = 6, depth = 0, use_2d = TRUE) {
  
  # vectors can be tensors for tensorflow, but we can also arrayify them to make them tensors; we need to 
  # do this because neattable (below) doesn't work with vectors
  if(is.null(dim(tensor))) {
    tensor <- array_reshape(tensor, dim = length(tensor))
  }
  
  # If we're not at the top level, we'll include an arrow character to indicate nesting
  # We'll also prefix the header line with the dimension name (should it have one) 
  prefix <- ""
  dimname <- ""
  seconddimname <- ""
  if(!is.null(dimnames(tensor)[[1]][1])) {
    possible_name <- dimnames(tensor)[[1]][1]
    if(possible_name != "[1]") { dimname <- possible_name }
    # if it's a 2d matrix, we want to print both names
    if(length(dim(tensor)) == 2 & !is.null(dimnames(tensor)[[1]][1])) {
      possible_name <- dimnames(tensor)[[2]][1]
      if(possible_name != "[1]") { seconddimname <- paste0(", ", possible_name) }
      dimname <- paste0(dimname, seconddimname)
    }
    
    if(depth > 0) {
      if(dimname != "") {
        prefix <- paste0("\U21B3 (", dimname, ") ")
      } else {
        prefix <- "\U21B3"
      }
    } else {
      prefix <- paste0("(", dimname, ") ")
    }
  } else {
    if(depth > 0) {
      prefix <- paste0("\U21B3")
    } else {
      prefix <- ""
    }  
  }
  
  
  # Print the header, indicating the rank of the tensor and the dimensions etc.
  cat(paste0(c(rep("    ", depth),
               prefix,
               "Rank ", 
               length(dim(tensor)), 
               " tensor, ",
               "[", 
               paste0(dim(tensor), collapse = ", "), 
               "]",
               ":\n"), 
             collapse = ""))
  
  # if the tensor is of rank 1 or 2 (ie vector or 2d matrix), we can just print it.
  # but, we use neattable (from the lgcp package) to print them with a given indent,
  # before printing 2d  we'll covert to a character tensor so we can chop off the right 
  # bottom, adding "..." entries to indicate missing rows/cols.
  rows_plus <- num_2d_rows + 1
  cols_plus <- num_2d_cols + 1
  
  if(length(dim(tensor)) == 1) {
    if(mode(tensor) == "numeric") {  # only show a few significant digits, but only if the mode is numeric
      tensor <- array_reshape(as.character(signif(tensor, digits)), dim = dim(tensor))
    }
    if(dim(tensor)[1] > n) { tensor <- tensor[1:n]; tensor[n] <- "\U22EF"} # EE
    
    neattable(tensor, indent = 4 * depth + 2) #  +2 for a little extra padding    
    
    
  } else if(length(dim(tensor)) == 2 & (any(as.logical(use_2d)))) {
    if(mode(tensor) == "numeric") {  # only show a few significant digits, but only if the mode is numeric
      tensor <- array_reshape(as.character(signif(tensor, digits)), dim = dim(tensor))
    }
    trimmedrows <- FALSE
    trimmedcols <- FALSE
    if(dim(tensor)[1] > rows_plus) { 
      tensor <- tensor[1:rows_plus, ]; 
      trimmedrows <- TRUE
      } 
    if(dim(tensor)[2] > cols_plus) { 
      tensor <- tensor[, 1:cols_plus]; 
      trimmedcols <- TRUE
    }
    # we convert to character *after* trimming, since the conversion can be very slow for large matrices.
    # Then we add dots to indicate trimming was done.
    if(trimmedrows) {
      tensor[rows_plus, ] <- "\U22EE";
    }
    if(trimmedcols) {
      tensor[, cols_plus] <- "\U22EF"; 
    }
    if(trimmedrows & trimmedcols) { tensor[rows_plus, cols_plus] <- "\U22F1"}
    
    neattable(tensor, indent = 4 * depth + 2) #  +2 for a little extra padding    
    # It's a 3-rank or higher; we're gonna do some recursion here
  } else {
    
    # We're going to list the sub-tensors, but if there are lots of them we're not going
    # to list all of them, but instead just print the first few
    if(dim(tensor)[1] > n) {
      
      # here's where it gets tricky: we need to grab the first couple of entries
      # from the first dimenions of the tensor. E.g. if dim(tensor) is c(10, 10, 10, 10), we 
      # want tensor[1:n, , , ]. But we can't write it that way because we don't know the rank to hard-code it.
      # The acorn() function (from the abind() package) can help with this, but it also
      # needs to take arguments like acorn(tensor, 2, 10, 10, 10). Fortunately we can use do.call()
      # to call a function with parameters generated from a list. 
      args_list <- list();
      
      # get the dimensions, build the argument list from the tensor and the dimension sizes
      tensor_dims <- dim(tensor)
      args_list[[1]] <- tensor
      for(i in 1:length(tensor_dims)) { args_list[[i+1]] <- tensor_dims[[i]] }
      
      # we only want the last n entries in the first dimenion (which is at index 2 of the param list)
      args_list[[2]] <- -1*n
      
      # make the call
      tensor <- do.call(acorn, args_list)
      cat(paste0(c(rep("    ", depth+1), "\U21B3 ... skipping ", tensor_dims[[1]] - n, " of ", tensor_dims[[1]]," rank ", length(dim(tensor))-1, " tensors ...\n"), collapse = ""))
      
      # Now, recurse. (See below for explanation of this line)
      invisible(apply(tensor, 1, array_print, n = n, digits = digits, num_2d_rows = num_2d_rows, num_2d_cols = num_2d_cols, depth = depth + 1, use_2d = use_2d))
      
      # Since in this case we truncated, we print a summary of the ones we're not printing
      
      # if there aren't too many, we can just do all of them
    } else {
      
      # If there are only a couple in the first dimension, no need to truncate, just recurse.
      # using apply() on the tensor in dimension 1 means we'll call array_print() on each entry
      # of the tensor, increasing the depth that it should be printed at.
      invisible(apply(tensor, 1, array_print, n = n, digits = digits, num_2d_rows = num_2d_rows, num_2d_cols = num_2d_cols, depth = depth + 1, use_2d = use_2d))
    }
  }
}

# returns a vector of dimension names, of the kind set by array_name_dims
# e.g. 
# named_rank3_tensor <- array_name_dims(rank3tensor, c("samples", "x", "y"))
# print(array_dim_names(named_rank3_tensor)) returns c("samples", "x", "y")
array_dim_names <- function(a) {
  return(unique(unlist(dimnames(a))))
}

# binds tensors to create another with a higher rank. 
# can take a list of tensors, or individual tensors
# optionally can take a new name for the new dimension
array_bind <- function(..., new_dim_name = NULL) {
  inputs <- list(...)[[1]]
  if(!is.list(inputs)) {
    inputs <- list(inputs)
  } 
  result <- abind(..., along = 0)
  
  if(!is.null(new_dim_name) & !is.null(dimnames(inputs[[1]]))) {
    oldnames <- array_dim_names(inputs[[1]])
    new_dimnames <- c(new_dim_name, array_dim_names(inputs[[1]]))
    result <- array_name_dims(result, new_dimnames)
  }
  return(result)
}





array_c <- function(...) {
  return(abind(..., along = 1))
}

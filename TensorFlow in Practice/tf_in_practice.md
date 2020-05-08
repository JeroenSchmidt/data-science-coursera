# Course 2
* When validation fluctuates alot while the test performance still increases 
 * This could indicate that our validation data does not have the same randomness that is present in the training data, this often happens when image augmentation is done excludisvly on the test set that is honomenues (eg like pose)

# Week 3: Transfer Learning 
Material:
* https://www.tensorflow.org/tutorials/images/transfer_learning

We can get cases where the validation score initially does well but then starts to perform worse as the model is trained longer. Overfitting is occuring. Use DROPOUT.

DROPOUT is a good thing to try when we see validation score diverging away from training score

Multiclass data gen, orders labels alphabetically


# Course 3

Word embeddings visualiser: 
https://projector.tensorflow.org/

Loss for text problems:
often we see, val_acc decreess but val_loss increases
intuitivly we can think of loss as confidence

>Think about loss in this context, as a confidence in the prediction. So while the number of accurate predictions increased over time, what was interesting was that the confidence per prediction effectively decreased. You may find this happening a lot with text data.

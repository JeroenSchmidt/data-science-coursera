# Course 1: Introduction to TensorFlow for Artificial Intelligence, Machine Learning, and Deep Learning


--------------------
# Course 2: Convolutional Neural Networks in TensorFlow
* When validation fluctuates alot while the test performance still increases 
 * This could indicate that our validation data does not have the same randomness that is present in the training data, this often happens when image augmentation is done excludisvly on the test set that is honomenues (eg like pose)

## Week 3: Transfer Learning 
Material:
* https://www.tensorflow.org/tutorials/images/transfer_learning

We can get cases where the validation score initially does well but then starts to perform worse as the model is trained longer. Overfitting is occuring. Use DROPOUT.

DROPOUT is a good thing to try when we see validation score diverging away from training score

Multiclass data gen, orders labels alphabetically

--------------------
# Course 3: Natural Language Processing Tensorflow

Word embeddings visualiser: 
https://projector.tensorflow.org/

Loss for text problems:
often we see, val_acc decreess but val_loss increases
intuitivly we can think of loss as confidence

>Think about loss in this context, as a confidence in the prediction. So while the number of accurate predictions increased over time, what was interesting was that the confidence per prediction effectively decreased. You may find this happening a lot with text data.

## Week 1: Sentiment in text

## Week 2: Word Embeddings

## Week 3: Sequence models

> Anacdotal from Laurence: I found from training networks that jaggedness can be an indication that your model needs improvement

> Aim for smoothness of loss & other metrics over epochs

> You tend to see more overfitting with text vs images because almost always you will have out of vocabulary words in the validation data set


## Week 4: Sequence models and literature (Text Generation)

**General outline**
 1. tokenize `from tensorflow.keras.preprocessing.text` `tokenizer.text_to_sequences`
 2. generate n-grams
 3. pad `tensorflow.keras.preprocessing.sequence.pad_sequences`
 4. make the last token the label we are predicting for 

**Note:** keras has a utility for onehot encodding (`tf.keras.utils.to_categorical`)

> Laurence: we can encounter out of memory problems when dealing with large (word) corpuses (eg shakespear), an alternative solution is to predict characters [docs example](https://www.tensorflow.org/tutorials/text/text_generation)

What about sparse categrocial classification?

--------------------
# Course 4: Sequences, Time Series and Prediction


**Appications:**
* Forcasting
* Imputation
* Anomaly detection
* Determine what generated the series itself (eg sound waves -> words)

## Week 1: Sequences and Prediction

**Common patterns & terms:**
* Trends
* Seasonality 
* Autocorreclation (lag relationships to past values)
* Inovations (cant be predicted on past values, i.e. random spikes)
* Noise
* None-stationary time series (e.g. when trends and seasonality change)

> Laurence: Maxim holds that more data is better if our TS is stationary, if that is note the case then we should focus on subsets that are stationary 

**Split Methodology:**

1. fixed partitioning
 * split data according fixed segments and ensure that each period contains a whole number of seasons.
 * e.g. You generally don't want one year and a half, or else some months will be represented more than others
2. roll-forward partitioning
    * same as fixed partitiong but roles it forward 

**Base Models**
* predict using previouse value
* moving averaging and differencing (predicting the TS delta and using that to predict the TS)
    * Forecasts = moving avg. of diff. series + series 
        * this will still have noise from the past reflected in our prediction
    * Forecasts = trailing moving avg of diff. series + centered moving avg of past series
        * So we can improve these forecasts by also removing the past noise using a moving average on that.
 
## Week 2: Deep Neural Networks for Time Series

**Creating and preparing Synthetic data using TF**
```python
# generate ranged data
dataset = tf.data.Dataset.range(10)
# truncates results
dataset = dataset.window(5, shift=1, drop_remainder=True)
# creates a "row"
dataset = dataset.flat_map(lambda window: window.batch(5))
#create rows of tuples
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
#shuffles row order
dataset = dataset.shuffle(buffer_size=10)
# creates batchs
dataset = dataset.batch(2).prefetch(1)


# alternative 
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
  dataset = tf.data.Dataset.from_tensor_slices(series)
  dataset = dataset.window(window_size + 1, shift=1, drop_remainder=True)
  dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
  dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
  dataset = dataset.batch(batch_size).prefetch(1)
  return dataset
```

> Sequence bias is when the order of things can impact the selection of things. For example, if I were to ask you your favorite TV show, and listed "Game of Thrones", "Killing Eve", "Travellers" and "Doctor Who" in that order, you're probably more likely to select 'Game of Thrones' as you are familiar with it, and it's the first thing you see. Even if it is equal to the other TV shows. So, when training data in a dataset, we don't want the sequence to impact the training in a similar way, so it's good to shuffle them up.

**Predicting for TS**
```python
for time in range(len(series) - window_size):
  #[np.newaxis] creates a temporary axis, this is done to reshape the input vector 
  #into a matrix that the model accepts
  forecast.append(model.predict(series[time:time + window_size][np.newaxis]))
```

## Week 3: Recurrent Neural Networks for time series

**univariance time series RNN**
![](figures/RNN-sequence-TS.png)

**Sequence to Sequence RNN**
* Two RNN layers where `return_sequences=True`
* TensorFlow assumes that the first dimension is the batch size, and that it can have any size at all, so you don't need to define it. Then the next dimension is the number of timestamps, which we can set to none, which means that the RNN can handle sequences of any length
* Keras useses the same dense layer independently at each time stamp
* We will however be disableing the last RNN sequence output
![](figures/RNN-2-TS.png)

**Lambda Layers**
```python
model = tf.keras.models.Sequential([
    # expand dims adds another dimension
  tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1),
                      input_shape=[None]),
  tf.keras.layers.SimpleRNN(40, return_sequences=True),
  tf.keras.layers.SimpleRNN(40),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 100.0)
])
```
* First layer helps us with dimensionality
    * If you recall when we wrote the window dataset helper function, it returned two-dimensional batches of Windows on the data, with the first being the batch size and the second the number of timestamps. 
    * But an RNN expects three-dimensions; batch size, the number of timestamps, and the series dimensionality. With the Lambda layer, we can fix this without rewriting our Window dataset helper function. 
    * Using the Lambda, we just expand the array by one dimension 
* Last layer helps us scale the predictions
    * The default activation function in the RNN layers is tan H which is the hyperbolic tangent activation. This outputs values between negative one and one. 
    * Since the time series values are in that order usually in the 10s like 40s, 50s, 60s, and 70s, then scaling up the outputs to the same ballpark can help us with learning
    
### Loss Functions

**Huber Loss**
In statistics, the Huber loss is a loss function used in robust regression, that is less sensitive to outliers in data than the squared error loss. A variant for classification is also sometimes used.

**Dynamic Learning Rate**
```python
from tf.keras.callbacks import LearningRateSCheduler
lr_schedule = LearningRateSCheduler(lambda epoch: 1e-8 * 10**(epoch/20))

model.fit(..., callbacks=[lr_schedule])
```

**Clearing TF Vars** important when using TF variables and lambdas
```python
tf.keras.backend.clear_session()  
```

## Week 4: Real-world time series data

**Using CNN**
```python
model = tf.keras.models.Sequential([
  tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
    #NOTE the input shape, this means we also have to edit the windowed_dataset function
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32, return_sequences=True)),
  tf.keras.layers.Dense(1),
  tf.keras.layers.Lambda(lambda x: x * 200)
])

def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    # we use the expand dims
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)
```

**Batch Size and Stability**
![](figures/loss_batch_size.png)
> From Laurance
* Some of the problems are clearly visualize when we plot the loss against the MAE, there's a lot of noise and instability in there. 
* One common cause for small spikes like that is a small batch size introducing further random noise. if you check out Andrea's videos and his course on optimizing for gradient descent,
    * One hint was to explore the batch size and to make sure it's appropriate for my data. So in this case it's worth experimenting with different batch sizes
# Run-Time Anomaly Detection for Software Systems

This machine learning project aims to reduce operational costs and improve the reliability of software systems by
detecting anomalies in their run-time behavior. The project is based
on [TU München's Enterprise Application Operations dataset](https://www.kaggle.com/datasets/anomalydetectionml/features)
and uses a variety of machine learning algorithms to detect anomalies in the run-time behavior of the software systems.

For the full project report, feel free to read my research paper, also available in this repository.

## TLDR

Deploying and managing software systems has become an increasingly hard job, especially in making sure that these
systems are reliable and secure. This research paper explores the problem of multivariate time series classification for
operational data from a real-world software system. The study compares the performance of two models, RNN with LSTM and
MLP, using limited windows of training data and multiple hyperparameter tuning. Holdout validation is
employed to test the accuracy of the models. The results demonstrate that RNN with LSTM significantly
outperforms MLP in terms of validation accuracy and, depending on the data sequencing approach for
MLP, even training time. Consequently, the study concludes that RNN with LSTM is the superior model
choice for this multivariate time series classification problem.

This project was conducted as the capstone project for the Neural Computing course at City, University of London,
achieving a distinction.

## The comparison

In this project, we compare the performance of two models, RNN with LSTM and MLP.
Recurrent Neural Networks (RNN) are a type of neural network that is designed to handle sequential data;
Long Short-Term Memory (LSTM) is a type of RNN that is capable of learning long-term dependencies.

Multilayer Perceptron (MLP) is a type of feedforward neural network that is commonly used for classification tasks.

Instead of using the entire dataset, we use a limited window approach that is similar to stream processing applications
such as Apache Flink that are commonly used in the industry to handle real-time data.

In this approach, we compare three different window size configurations:

1. **Tumbling window**: A fixed-size window that does not overlap with the next window.
2. **Sliding window**: A fixed-size window that overlaps with previous and next windows.
3. **Increasing window**: A window that increases in size with each new data point.

## The dataset

The dataset was taken from Kaggle and is provided by TU München. It contains operational data from a real-world
enterprise application.
For this project, only ~5% of the dataset (or 188058 rows) was used due to the limited computational resources
available.
The dataset contains 235 columns, with the first 234 columns being features and the last column being the label.

The target variable was created based on system activity preceding a node restart.

The following steps were taken to preprocess the data for our experiments:

1) **Missing values**. Although the used dataset does not contain missing values, it is important to note that the
   original, unprocessed dataset did in fact have missing values. Huch et al. replaced missing values using
   linear interpolation to retain the information and allow machine learning algorithms to work with the data. The No
   data column resembles this situation, therefore we remove this artificial column.
2) **Label encoding**. To enable the models to work with the data, we label encoded the target variable.
3) **Encoding categorical variables**. We transformed the categorical variables, i.e. the host and the process
   columns by using one-hot encoding for an integer representation.
4) **Handling time**. To work with the timestamps, we converted the datetime into UNIX timestamps that can be
   represented by integers. We then sorted the data by time.
5) **Holdout validation split**. Holdout validation was used for our experiments. We used an 80-10-10 ratio for
   splitting the data into a training, validation, and test set.
6) **Highly correlated values**. The dataset contains quite a few correlated features. We removed
   features with high correlation and high anti-correlation to further increase model performance. For our
   experiments, we took 0.95 as the removal threshold.
7) **Constant values**. We identified columns that contain just constant values, i.e. columns that have one as the
   number of unique values and removed them to increase model performance.
8) **Rebalancing the dataset**. In our case, only 16.4% of the target variables belong to the minority class.
9) **Scaling**. Numerical features in the dataset differ in their value range. As scaling the dataset tends to increase
   model performance, we normalized all numerical features fi so that fi = fiσ−µ where µ is
   the mean and σ is the standard deviation. We scaled the training set, validation set, and test set separately.
   Further processing was then applied for model specifications.

## Tuning hyperparameters

### RNN with LSTM

The model consists of 1 and later 2 layers of LSTM cells connected to a linear output layer.
The model uses the gradient stochastic descent with RMSprop as the optimizer and cross-entropy as the loss function.

For tuning the RNN model, we decided to focus on the following hyperparameters:

1) Learning rate
2) Weight decay
3) Hidden layer size
4) Number of LSTM cell layers

Since we used gradient stochastic descent with RMSprop which has an adaptive learning rate, we did not tune momentum.

### MLP

The model consists of an input layer, a tanh activation function, another layer, tanh activation, and finally the output
layer with a softmax activation function.

The model uses the gradient stochastic descent with RMSprop as the optimizer and cross-entropy as the loss function.

For tuning the MLP model, we decided to focus on the following hyperparameters:

1) Learning rate
2) Weight decay
3) Hidden layer size
4) Momentum

## Results

The results of tuning RNN with LSTM revealed several key findings.

Firstly, reducing the number of layers
from 2 to 1 led to a significant reduction in training time, but also caused a slight decrease in accuracy. Secondly,
incorporating a dropout rate resulted in an improvement in model performance. Thirdly, introducing weight decay
caused a drastic decrease in model performance. Finally, the best RNN model was achieved with 2 layers, a
hidden size of 100, a learning rate of 0.0025, no weight decay or L2 regularization, no momentum, and a dropout
rate of 0.4.

These findings suggest that careful tuning of hyperparameters can lead to significant improvements
in RNN model performance for multivariate time series classification tasks. Validation accuracy was between
0.65 and 0.9 for most of the experiments, showing promising results, while the training cost was 15 seconds to
40 seconds.

The results of the MLP experiment revealed that the performance of the MLP models varied significantly
depending on the windowing strategy used. Sliding windows consistently yielded the worst results, while
tumbling windows were marginally better.

Although increasing windows produced the most promising results,
they required significantly greater computational efforts, making them more expensive to use.

Despite this, even
the best-performing MLP model with increasing windows had significantly lower accuracy than the RNN/LSTM
models. The optimal MLP model had a hidden size of (50, 100, 100), a learning rate of 0.003, no weight
decay/L2 regularization, and no momentum. 

Results were inconsistent though, since MLP weights and biases
are randomly initialized, making them prone to fluctuation in performance and local minima. MLP validation
accuracy consistently was between 0.60 and 0.75, showing its limited ability to capture temporal relationships.
Training time using GPU, on the other hand, was between 15 seconds and 120 seconds, depending on the
windowing strategy.

There are several factors that may explain the results observed in this study.
Firstly, it is well-established that RNN with LSTM is particularly well-suited for handling sequential data,
such as time series data. This is due to the memory cells in the LSTM layer that can selectively ”remember”
important information from previous time steps, allowing the model to capture temporal dependencies in the
data. 

In contrast, MLPs are not inherently designed to handle sequential data and may struggle to capture the
complex patterns in multivariate time series data.
Regarding the MLP results, the fact that the performance varied significantly depending on the windowing
strategy used highlights the importance of careful preprocessing and feature engineering in machine learning
tasks. 

It is possible that the sliding window strategy may have introduced noise or irrelevant data into the model,
leading to poorer performance. It is also worth noting that MLPs are typically less well-suited to handle large
amounts of data than RNNs, which may explain why the best-performing MLP model had lower accuracy than
the LSTM models. Finally, the fact that MLP weights and biases are randomly initialized may explain the
inconsistent results observed in this study, as the model may have converged to different local minima depending
on the initialization.

## Remarks

The project was optimized for running on GPU in Google Colab.

The data could not be included in this repository due to its size. However, the data can be downloaded from the
the link provided above and should be put into a dedicated "data" folder to seemlessly run the code.
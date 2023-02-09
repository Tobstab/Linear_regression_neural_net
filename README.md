# Linear_regression_neural_net
## Part 1 - Data Pre-Processing

Before beginning the actual implementation of the Multi-layer Perceptron algorithm, I believed that data pre-processing should take precedence. So, I began by cleansing the data first, for this I needed to understand the data, so that I would know what seemed appropriate and what the possible outliners would be. This is important as data that contains outliers or errors, will produce incorrect trends and correlations, which means that when the neural network is provided new predictors, its prediction will be completely incorrect.
So, I began by comparing each predictor against the predictand using excel, this was used as it very producing graphs to show correlations and trends. I first compared the Area agaisnt the Index flood, doing this showed me that there was value that came out as -999, this was clearly an outlier as an area cannot negative. This fact was true for all predictors, so I knew I needed to look out for negative numbers. I created graphs for each predictor and discovered that they too contained negative numbers.

![image](https://user-images.githubusercontent.com/9751439/217943179-e9fe9c12-7587-4869-b54d-30925cbdd276.png)

I decided to create an algorithm to perform the cleanse – this function was named the ‘prep_data’ function. I decided to create an automatic cleansing function, so that the whole system could be repurposed to be used on a different dataset without having to look at it properly. To do this, I brought dataset in as a text file, splitting each line into row of data to be appended into an array called the ‘dataset’. Then for each row in the array I performed calculations in order to look out for negative numbers, if found the ‘prep_data’ function removed the entire row. When I first ran the code to cleanse the data, I was unable to process the input due to the function not being to parse through strings, which meant the data had more errors and outliers. So, I had a check for non-floats /integers into the system, if any were found once again the row was completely deleted. This meant that the ‘prep_data’ function was able to return a cleansed dataset, with no errors.
The next part was making sure the data points were standardised. This is 
important step as this makes sure that data is internally consistent, and it also means reduces the size of the delta values when updating the weights, which means the resultant model will be more stable. To do this, I needed to get the maximum and minimum values of all the predictors and the predictand. So for each column, used the in built max() and min() functions to find the min and max values. Using these values, I calculate the standardised values for each data point. I decided to standardise my values between the ranges 0.1 and 0.9, using the formula below. This was done to help with the prediction of the model, so it would be able best produce values outside the scope of the training data predictors and would not be as subject to over training.

![image](https://user-images.githubusercontent.com/9751439/217943325-dbc01607-9b62-46ce-ae70-c6d3b76b6f61.png)

Finally, I split the dataset into three sets, the training set, the validation set and the testing set. This was decided by giving the training set 60% of the data, and the other two were given 20% each.

## Part 2 – implementation of gradient descent

I decided to use a procedural approach when creating the system. So, I first began by creating a function to initialise a network model. The outcome of this function was to return all the weights of the model, all structured into individuals’ layers and their own nodes. So, for this, the parameters the function needed were a number of inputs and a number of hidden nodes. So, for example with the inputs (8,4) the function produced a three-dimensional array with two layers, the first layer containing 4 node arrays, each with 9 weights associated to them – 8 for input and 1 for the bias value. The second layer then would contain one node, with weights from the hidden nodes to the output node including the bias for the output node. Each weight was initialised with random weights between -1 and 1.
After creating the network, the next task was to implement the forward propagation algorithm, so that inputs could be fed into the network in order to produce an output. This function – named ‘forward_prop’, needed the model of the network and a row of data as parameters. The main algorithm involved creating an output array which stored the results from each layer. And for each node in a layer, the function computed sum of all the weights for that node agaisnt the predictands. After running the first layer through the algorithm, it would produce all the outputs for the hidden nodes, using the inputs from the dataset. The second layer would produce a single value of the output in the output layer using the outputs from the hidden layer as inputs. I also created a ‘node_activation’ function which took in the outputs for each node and performed a sigmoid function on each so that values can be match to a predictand. The ‘forward_prop’ function would then later be implemented as part of the backpropagation algorithm aswell for predicting new index floods later for the validation set and testing set.

I then began to work on the backpropagation function, this function needed many parameters - these being the initial model of randomised weights, the training set, the learning rate, and the number of epochs. The initial way I implemented the backpropagation algorithm, was by taking each row of data and computing derivatives for each predicted output to each layer from the forward pass. Then I calculated the cost function for each node in the two layers. I then calculated new weights using the learning rate and the cost functions. The cost function was computed using the gradient descent method, which means calculating deltas values for each predicted output against the actual output. This meant for a training set of 100 data rows, weights would be updated every row of data per epoch. So already at this point I could see how the performance of the system may be impacted by many epochs. In this function I also computed a root mean squared error, so that I would be able to see the error for each epoch and monitor the performance. Then using the Matplotlib library I added each error per epoch to an error so I could plot the end results.
![image](https://user-images.githubusercontent.com/9751439/217943839-2134aee6-7f5a-412e-b687-4c5af547ee7f.png)

Another vital part of the system was a function that allowed me to test the network agaisnt the validation set and the testing set. This was a simple implementation; it combined the root mean error squared computation from the backpropagation algorithm aswell as the forward prop function in order to produce outputs using the new weights and biases from the improved network model. I also calculated a percentage error for each network, so that I two measures of model accuracy. For percentage error, the main goal was to find any model under 15% and for the root mean squared error, any value under 0.04 tended to be a hopeful model. This function also uses matplotlib in order to produce a graph of the percentage error between each correct output and the predicted value.
Finally, I created a model store function, that allowed me to save models that were below a certain threshold. This function stored the learning rate, number of hidden nodes, root mean squared error and the percentage error of the system. The threshold I had selected was a percentage error rate of 15 or less.
The graphs above were produced using a learning rate of 0.1 and 1000 epochs.
![image](https://user-images.githubusercontent.com/9751439/217943864-e1b24736-0432-4ad0-968a-a79b9a583a42.png)

## Part 3 – improvements

Whilst testing for improvements I had the model store threshold set to 40% so that I could see how the improvements changed over time. Aswell for each improvement, weights were kept constant throughout each iteration to make sure the weights did not produce anomalies and outliers. This was done by using a while loop around the validation function so the same initial network was passed through but values such as the learning rate, number of hidden nodes and epochs could be changed each time.
Changing number of epochs

The first improvement that I wanted to implement was how number of epochs affected the percentage error and improvement rate of the network. So, I kept a constant learning rate of 0.1 and 4 hidden nodes and began to check the performance of the network. I increase the epoch amount by 100 each time, I decided to pick a threshold of 2000 epochs as anymore would have a large runtime, with the current backpropagation algorithm.
I noticed instantly that a lower number of epochs impacted the actual performance of the network, a lower number of epochs, led to a quicker runtime than a higher number of epochs.
While run different models of the network with different number of epochs, I discovered that on average a larger number of epochs led to greater reduction in percentage error. This was expected as more cycles through the data would allow for more adjustments to the weights and therefore would improve the network overall.
As you can see from the graph below an epoch rage of 1000 to 1500 seemed to have the greatest improvement on the network. The lowest percentage error being 12.8% at an epoch of 1300.

# See improvements on models graph for the rest of the improvements

# See Complete report for comparisons of different neural network techniques

## Part 5 – conclusion
The table shows the comparisons between the 6 best improvements that I had found. From the table below the best percentage error found was within the model an increased learning rate using the sigmoid activation function. While the model with the smallest root mean squared error was the Normal Tanh-1 model.

![image](https://user-images.githubusercontent.com/9751439/217945819-c3326790-e69d-4306-a0f9-cecc9cd18612.png)

With the sigmoid model not being too far away in error rate. I expected the other model improvements to have the biggest impact on percentage error and error rate due to them being able to find the global minima the fastest. However, these results indicate that perhaps a little over training began to occur during the large number of epochs.
Using the results, that I have gathered, I concluded that the best model is the sigmoid model with a learning rate of 0.4 and 9 hidden nodes.


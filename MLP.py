'''
----------------------------------
Artifical Neural Network
created to help predict the index flood
using pre-found data
created by Tobi Akinyemi B921352
----------------------------------
'''

import random
import math
import matplotlib.pyplot as plt
import ast

#Data Preparation Section
def prep_data(dataset): #function to prepare the data, by removing anomalies
    raw_data = []
    with open(dataset, 'r') as data: # open the text file containing the dataset
        for line in data.readlines(): 
                raw_data.append(line.strip('\n').split('\t'))# add each line to a python array
        for row in raw_data: # for each row in the raw_data array
            for item in row: # for each data point in each row
                #get rid of data points that aren't types of numbers
                try:
                    float(item)
                except ValueError:
                    raw_data.remove(row)
        for row in raw_data:
            for item in row:
                #get rid of values that are negative
                if float(item) < 0:
                    raw_data.remove(row)
    return raw_data
                    
def standardise(pre_clean_set): #fucntion for standardising the data
    mins= [] #array for the minimum values of each column
    maxs = [] #array for the maximum values of each column
    for i in range(len(pre_clean_set[0])): # for each data row
        column =[]
        for j in range(len(pre_clean_set)): #for each value in a column
            column.append(float(pre_clean_set[j][i]))#add each value to a column array
        mins.append(min(column))#get minimum value of each column
        maxs.append(max(column))#get maximum value of each column
    for i in range(len(pre_clean_set[0])): #for each data row
        for j in range(len(pre_clean_set)): #for each data point
            pre_clean_set[j][i] = round(0.8*((float(pre_clean_set[j][i]) - mins[i])/(maxs[i] - mins[i]))+0.1, 3) #set values to be between 0.1 and 0.9

    #split into training set, validation set and testing set.
    training_set =[pre_clean_set[i] for i in range(round(len(pre_clean_set)*0.6))] #get 60% of the data for the training set
    validation_set =[pre_clean_set[i] for i in range(round(len(pre_clean_set)*0.6),round(len(pre_clean_set)*0.8))] #get 20% of the data for the validation set
    testing_set =[pre_clean_set[i] for i in range(round(len(pre_clean_set)*0.8), len(pre_clean_set))] #get 20% of the data for the training set
    return (training_set, validation_set, testing_set) #return the different data sets

#training section          
def ML_network(training_set, num_hidden):
    network = [] #the network is stored in a 3 Dimensional array
    change=[] #this will be the array that stores the previous weights same form as the network array
    hidden_layer =[]#stores the weights of the hidden layer
    hidden_change_layer =[] # hidden layer for the change network
    
    num_inputs = len(training_set[0])#get number of inputs
    
    for i in range(num_hidden): #for each number of hidden layer
        hidden_layer.append([round(random.uniform(-2/num_inputs-1,2/num_inputs-1 ),2) for i in range(num_inputs)]) #assign a weight per predictand, where predictands is num_inputs
        #weights set to value between -2/n and 2/n
        hidden_change_layer.append([round(random.uniform(0,0),2) for i in range(num_inputs)]) #set all weights to 0 initailly
    change.append(hidden_change_layer)#add the hidden change layer to its network
    network.append(hidden_layer)#add the hidden layer to the network

    num_hiddens = num_hidden + 1 #assign the number of weights the hidden layer has to the output node, +1 for bias
    network.append([[round(random.uniform(-2/num_inputs-1,2/num_inputs-1),2) for i in range(num_hiddens)]])#for the number of hidden node weights assign random weights
    change.append([[round(random.uniform(0,0),2) for i in range(num_hiddens)]])#same as step above for the change network
    return (network,change)#return both networks

def node_activation(activation_sum): #function for activating nodes after getting sums
    activated_node = 1/(1 + math.exp(-activation_sum)) #uses sigmoid fucntion for activation
    return activated_node

def node_activation2(activation_sum):
    activated_node = (math.exp(activation_sum)-math.exp(-activation_sum))/( math.exp(activation_sum) + math.exp(-activation_sum))
    #uses tanh-1 as activation function
    return activated_node

def forward_prop(network, data_row):#function for propagating forward in the network
    predictors = data_row #get a row of inputs
    outputs =[] #empty list of outputs that provided at the end of each propagation
    for layer in network: #for each layer in the network
        hidden_to_output =[] #array for transferring outputs to the next layer
        for node in layer: # for nodes in each layer
            activation_sum=node[-1]#add the bias to the sum of activations
            for i in range(len(node)-1):
                activation_sum+=predictors[i] * node[i]#then add the input * weight for each input to the node
            hidden_to_output.append(node_activation(activation_sum))# activate the node and add it to the list to transfer between layers
            outputs.append(node_activation(activation_sum)) #only useful for final output, assign last outputs as hidden to output is cleared everytime
        #print(hidden_to_output)
        predictors= hidden_to_output#transfer to next layer
    return outputs #return all outputs

def back_prop(network,previous_weights, training_set, lr, epochs):# function for backpropagating through network
    errors=[] #errors per epoch
    print(network) #output the current state of the network
    #simulated annealing parameters
    e_param = 0.01 #learning rate at the end
    s_param =0.1 #learning rate at the start
    for i in range(epochs): #for eahc epoch
        #simulated annealing equation
        lr = e_param +(s_param -e_param)*(1-(1/(1+math.exp(10-((20*i)/epochs)))))
        error=0#current error within epoch, i
        previous_network=previous_weights # assign the values from the previous network to this function
        for i in range(len(training_set)): # for each row in the dataset
            output = forward_prop(network, training_set[i])
            inputs=[training_set[i][j] for j in range(len(training_set[i])-1)] #add inputs to an array
            inputs.append(1)# add bias input
            #calcalute deltas
            #d_output = [output[i]* (1 - output[i]) for i in range(len(output))] #used for sigmoid
            d_output = [1 - ((output[i])**2) for i in range(len(output))] #used for tanh-1
            delta_0 = (training_set[i][-1] - output[-1]) * d_output[-1] #calculate delta 0
            deltas = [] # array for all deltas
            layer=network[1] #only calculate deltas for hidden layer
            for node in layer: # for node in the layer
                for i in range(len(node)-1):
                    deltas.append(node[i]*delta_0*d_output[i]) 
            deltas.append(delta_0)
            error += ((training_set[i][-1] - output[-1])**2) /len(training_set) #sum together error for this epoch
            
            #update weights
            i=0
            for layers in network:
                if layers != network[-1]: #only do this for hidden layer
                    for nodes in layers:
                        for j in range(len(nodes)):
                            diff_weight =previous_network[network.index(layers)][layers.index(nodes)][j] # get the difference between last weight
                            nodes[j] += lr * deltas[i]*inputs[j] +(0.9*diff_weight) #update the weights for hidden layer includes momentum
                            previous_network[network.index(layers)][layers.index(nodes)][j]=previous_network[network.index(layers)][layers.index(nodes)][j] - nodes[j]
                            #change difference to new difference
                            
                        i+=1
                else:
                    output[-1]= 1
                    for nodes in layers: #only do this for output layer
                        for j in range(len(nodes)):
                            diff_weight =previous_network[network.index(layers)][layers.index(nodes)][j]
                            nodes[j] += lr * deltas[i]*output[j] +(0.9*diff_weight) #update the weights for output layer includes momentum
                            previous_network[network.index(layers)][layers.index(nodes)][j]=previous_network[network.index(layers)][layers.index(nodes)][j] - nodes[j]
                    
                        i+=1
        errors.append(error)#add the error of the epoch to the errors array
        #bold driver improvement
        if (error> errors[-1]*1.04) and i%100 == 0: #if error increases by 4% then decrease lr by 30%
            lr =lr*0.7
        else:
            if lr < 0.5: #if learning rate decreases then increase the lr by 5%
                lr=lr*1.05
        
       
    plt.plot(errors) #plot the errors
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.show()
    print(network) #show the new update network
    return(network)

def batch_back_prop(network,previous_weights, training_set, lr, epochs): #attempted batch processing
    #same as above back propagation but weights updated at the end of epoch
    errors=[]
    for i in range(epochs):
        error=0
        previous_network=previous_weights
        parameters=(len(network[-1][0]))
        d_grads = [0 for i in range(parameters)]
        layer=network[1]
        for x in range(int(len(training_set))):
            for y in range(len(training_set[0])):
                output = forward_prop(network, training_set[x])
                inputs=[training_set[x][y] for y in range(len(training_set[x])-1)]
                inputs.append(1)
                #calcalute deltas
                d_output = [output[i]* (1 - output[i]) for i in range(len(output))]
                delta_0 = (training_set[x][-1] - output[-1]) * d_output[-1]
                error += ((training_set[x][-1] - output[-1])**2) /len(training_set)
                output[-1]= 1
                for node in layer:
                    for k in range(len(node)-1):
                        for j in range(len(inputs)):
                            d_grads[k]+=(node[k]*delta_0*d_output[k]*inputs[j])
                for j in range(len(output)):
                    d_grads[-1]+=(delta_0 *output[j])
                
            #update weights
        errors.append(error)
        for layers in network:
            for nodes in layers:
                for j in range(len(nodes)):
                    diff_weight =previous_network[network.index(layers)][layers.index(nodes)][j]
                    nodes[j] += lr * (d_grads[layers.index(nodes)]+ (0.9*diff_weight)) *(1/len(training_set)) 
                    previous_network[network.index(layers)][layers.index(nodes)][j]=previous_network[network.index(layers)][layers.index(nodes)][j] - nodes[j]
                
            
    plt.plot(errors) 
    plt.show()
    return(network)

def validation(network,validation_set): #fucntion for evaluating the models
    pred_results =[] #list of the predict
    error_sqrd=0 #error for root mean squared error
    error=0 #error for percentage error
    for i in range(len(validation_set)):
        output = forward_prop(network, validation_set[i]) #calculate output layer output
        error_sqrd+=(validation_set[i][-1] - output[-1])**2 #error squared for each row of validation/training data
        error+=abs(validation_set[i][-1] - output[-1])/validation_set[i][-1]
        pred_results.append(((output[-1] - validation_set[i][-1] )/ 1)* 100) 
    rmsq = round(math.sqrt(error_sqrd/len(validation_set)),4)#root mean squared error
    p_error = round((error/len(validation_set))*100,2) #percentage error for each prediction
    print("root mean squared error: ",rmsq)
    print("Percentage Error: ",p_error)
    plt.plot(pred_results) #plot the results
    plt.xlabel("Row of data")
    plt.ylabel("percentage error")
    plt.show()
    if p_error < 15: #if the model has a percentage error less than 15 then it should be saved
        model_store(network, p_error, rmsq)
    display_pred =input("Would you like to see the predicted agasint the actual outcome? ")
    if display_pred.lower() == "yes": #for displaying the training data agaisnt the predicted model
        show_pred(network,standardises_sets[2])
        
    

def model_store(network,p_error,rmsq): # function for saving the models
    save =input("Would you like to save this model?: ")
    if save.lower() == "yes":
        lr = input("please enter the lr: ")
        h_nodes = input("please enter the number of hidden nodes: ")
        epochs = input("please enter the number of epochs: ")
        with open('savedModels.txt','r+') as f: #reads the file adds it into the file
            line = f.readlines()
            lines = len(line)
            model_num = int(lines/3 +1)#calculates model number based on number of previous models
            f.write(("model "+str(model_num)+" rmsq: "+str(rmsq)
                     +" p error: "+str(p_error)+ " epochs: "+str(epochs)+"\n"))
            f.write(("hidden: "+h_nodes+" lr: "+lr+"\n"))
            f.write((str(network)+"\n"))
    else:
        print("model not saved")
    
def run_model(): #function for running the models
    network = ast.literal_eval(input("input model: ")) #input in a model
    validation(network,standardises_sets[1])#runs the model agaisnt the validation model 

                
    
def show_pred(network,training_set): #shows the predicted neural network model, linest model and actual values
    preds=[]
    outputs=[]
    linest_pred =[15.86427869,58.12542093,94.59202479,49.90393272,34.64427658,195.4741057,67.03019826,64.72755454,64.20768892,283.5274624,112.5516162,8.285051503,81.00879861,44.54792807,226.5799856,61.42550401,-17.94422686,60.79736232,-51.14630838,148.4609947,75.4560316,84.13015751,124.2643131,70.26273895,123.0008176,139.9182192,300.0924499,63.85528415,-20.26821419,151.8763799,50.92907703,-7.575861102,54.95308624,-31.36135872,2.065822498,4.518317461,27.87135714,105.5573693,173.7538024,65.15668052,42.44629094,72.84285217,26.04503095,267.3210548,-37.9175065,79.86747085,104.3426847,-23.43462627,155.0900214,39.76489359,129.1110319,132.0823108,113.0099745,172.6379297,213.9043697,111.5060309,140.6919372,314.7900825,544.877498,114.9306304,108.853313,84.14710762,155.2099642,111.472885,15.01801188,102.7282959,43.23201382,135.5290351,199.5644843,-65.94988532,40.95559208,-42.97923523,98.47326739,41.38950229,132.4855993,39.34120142,30.01771328,80.54518532,-64.12975148,50.12628332,29.01388474,-48.46892693,78.73250874,51.76068626,375.0448333,113.5702457,162.6154507,158.070053,58.27909104,136.5263133,73.16663497,89.75015865,96.95179884,34.01463385,74.37181059,31.49287919,85.15792679,693.6141057,52.69655398,75.25070114,102.5389648,259.1697642,81.7514564,13.4338133,96.1696078,44.11711561,21.9293656,8.882152976,36.85488016,79.31604214,-26.32528803,-36.50100198,15.72226717,49.12711926,51.85853205,62.25414245,262.7627493,126.0800402]
    #above are the linest values
    for i in range(len(training_set)):
        pred = forward_prop(network, training_set[i])
        dstandard_output=(((training_set[i][-1]-0.1)/(0.8))*(992.846-0.142))+0.142 #destandardise actual output
        dstandard_pred=(((pred[-1]-0.1)/(0.8))*(992.846-0.142))+0.142 #destandardise the predicted values
        preds.append(dstandard_pred)
        outputs.append(dstandard_output)
    plt.plot(outputs, label ="actual") #plot the graphs
    plt.plot(preds, label ="predicted")
    plt.plot(linest_pred, label ="linest Prediction")
    plt.xlabel("Row of data")
    plt.ylabel("Flood Index")
    plt.legend()
    plt.show()

#main code to run everything. By commenting out lines below some functions can be prioritised.    
standardises_sets=standardise(prep_data("DataSet.txt"))#standardises dataset
net=ML_network(standardises_sets[0], 9)#creates network based on number of hidden nodes
validation(back_prop(net[0],net[1],standardises_sets[0],0.1,150),standardises_sets[1]) #validates the model, after training it
#based off number of epochs, learning rate
run_model()# runs the models that inputted for testing.
    


    



    

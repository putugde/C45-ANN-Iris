import pandas as pd
import math

def sigmoid(inputs, perceptron_weight):
    net = 0

    for i in range(len(inputs)):
        net += inputs[i] * perceptron_weight[i]

    # (1/(1 + e^-x))
    output = 1/(1+(math.pow(math.e, -net)))

    return output



def mini_batch_gradient(df,attribute,target_attribute):
    learning_rate = 0.1
    target_value = []
    perceptrons = [4,4,3] # jumlah perceptron di setiap layer
    weight = [[],[]]
    '''
    weight/delta_weight : 3d matriks, weight[i][j][k] = weight dari perceptron j di layer i, ke perceptron k di layer i+1
    '''
    delta_weight = [[],[]]
    batch_size = 15
    epoch = 1000

    # initiate weight and delta weight with 0
    for i in range(len(perceptrons)-1):
        for _ in range(perceptrons[i]+1):
            weight[i].append([0 for _ in range(perceptrons[i+1])])
            delta_weight[i].append([0 for _ in range(perceptrons[i+1])])
    
    
    for _,row in df.iterrows():
        if row[target_attribute[0]] == 'Setosa':
            target_value.append([1,0,0])
        elif row[target_attribute[0]] == 'Versicolor':
            target_value.append([0,1,0])
        elif row[target_attribute[0]] == 'Virginica':
            target_value.append([0,0,1])
    
    for ep in range(epoch):
        print('\n\n EPOCH '+str(ep)+'\n')
        current_batch_counter = 0
        for i,row in df.iterrows():
            # print(i)
            input_perceptron = [[1,row['sepal_length'],row['sepal_width'],row['petal_length'],row['petal_width']]] 
            '''
            input_perceptron : matriks 2d, input[i][j] = nilai input perceptron j di layer i 
            '''
            
            for layer_idx in range(len(perceptrons)-1):
                input_perceptron.append([1]) # bias
                for perceptron_output_idx in range(perceptrons[layer_idx+1]): # indeks buat dapatin weight ke perceptron target
                    perceptron_weight = [] # semua weight yang masuk ke perceptron target
                    for perceptron_input_idx in range(perceptrons[layer_idx]+1): # loop buat dapatin indeks perceptron asal
                        perceptron_weight.append(weight[layer_idx][perceptron_input_idx][perceptron_output_idx]) 
                    output = sigmoid(input_perceptron[layer_idx], perceptron_weight) # aktivasi
                    input_perceptron[layer_idx+1] += [output] # nilai perceptron baru  yang akan digunakan sebagai input di next forward
                    # print(input_perceptron[layer_idx+1])
            # print(weight)
            prediction_output = input_perceptron[-1][1:] # remove bias from output
            total_output = sum(prediction_output)
            prediction_output = [x/total_output for x in prediction_output]
            # print(prediction_output)
            current_batch_counter += 1
            # update delta weight (?)
            
            # hanya untuk 1 hidden layer
            for layer_idx in range(len(perceptrons)-1, 0, -1):
                for perceptron_output_idx in range(perceptrons[layer_idx]):
                    for perceptron_input_idx in range(perceptrons[layer_idx-1]+1):
                        if layer_idx == len(perceptrons)-1:
                            weight_change = input_perceptron[layer_idx][perceptron_output_idx+1]-target_value[i][perceptron_output_idx]
                            weight_change *= (input_perceptron[layer_idx][perceptron_output_idx+1])*(1-input_perceptron[layer_idx][perceptron_output_idx+1])
                            weight_change *= input_perceptron[layer_idx-1][perceptron_input_idx]
                            delta_weight[layer_idx-1][perceptron_input_idx][perceptron_output_idx] += weight_change
                        else:
                            total_ET_OH = 0
                            for output_idx in range(perceptrons[len(perceptrons)-1]-1):
                                cur_et_oh = input_perceptron[len(perceptrons)-1][output_idx+1]-target_value[i][output_idx]
                                cur_et_oh *= input_perceptron[len(perceptrons)-1][output_idx+1]
                                cur_et_oh *= (1-input_perceptron[len(perceptrons)-1][output_idx+1])
                                cur_et_oh *= weight[1][perceptron_output_idx+1][output_idx]
                                total_ET_OH += cur_et_oh
                            weight_change = total_ET_OH * input_perceptron[layer_idx][perceptron_output_idx+1]
                            weight_change *= (1-input_perceptron[layer_idx][perceptron_output_idx+1])
                            weight_change *= input_perceptron[layer_idx-1][perceptron_input_idx]
                            delta_weight[layer_idx-1][perceptron_input_idx][perceptron_output_idx] += weight_change

            if current_batch_counter == batch_size:
                # update weight with delta weight
                for i in range (len(perceptrons)-1):
                    for j in range (perceptrons[i]+1):
                        for k in range (perceptrons[i+1]):
                            weight[i][j][k] -= learning_rate*delta_weight[i][j][k]
                            delta_weight[i][j][k] = 0
                current_batch_counter = 0
            # done
            if ep == epoch-1:
                print(str(prediction_output)+' <-- harusnya --> '+row[target_attribute[0]])

if __name__ == '__main__':
    url = 'iris.csv'
    data = pd.read_csv(url)
    attributes = ['sepal_length','sepal_width','petal_length','petal_width']
    target_attribute = ['variety']
    mini_batch_gradient(data,attributes,target_attribute)
    # assume one hidden layer with 4 neuron 





# neuron layer:
# 4 -> 4 -> 3
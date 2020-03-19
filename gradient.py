import pandas as pd
import math
import random


random.seed(1)

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
    epoch = 186
    maximum = -1
    epoch_max = -1

    # initiate weight and delta weight with 0
    for i in range(len(perceptrons)-1):
        for _ in range(perceptrons[i]+1):
            weight[i].append([0 for _ in range(perceptrons[i+1])])
            delta_weight[i].append([(random.random()*2)-1 for _ in range(perceptrons[i+1])])
    
    result = ['Setosa','Versicolor','Virginica']

    prediction_result = []

    for _,row in df.iterrows():
        if row[target_attribute[0]] == 'Setosa':
            target_value.append([1,0,0])
        elif row[target_attribute[0]] == 'Versicolor':
            target_value.append([0,1,0])
        elif row[target_attribute[0]] == 'Virginica':
            target_value.append([0,0,1])
    
    for ep in range(epoch):
        print('EPOCH '+str(ep+1))
        accuracy = 0
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
            # if ep == epoch-1:
            result_index = prediction_output.index(max(prediction_output))
            if ep == epoch-1:
                prediction_result.append(result[result_index])
            if (result[result_index] == row[target_attribute[0]]):
                accuracy += 1
                # print(f'Prediction: {result[result_index]}\nReality: {row[target_attribute[0]]}\n\n')
        # if ep == epoch-1:
        if(accuracy > maximum):
            maximum = accuracy
            epoch_max = ep + 1
        # print(f'accuracy = {accuracy}/150\n')

    # print(f'\n\nepoch max : {epoch_max}\naccuracy = {maximum}/150')
    # print(weight)
    # for i in range(len(weight)):
    #     print(f'\n\nlayer ke-{i+1}')

    #     for j in range(len(weight[i])):
    #         for k in range(len(weight[j])):
    #             print(f'{j}-{k} : {weight[i][j][k]}')
    return prediction_result

def createConfusionMatrix(result, oldData, target_attribute):
    # order : [[true positive, false positive, false negative, true negative]]
    # order : [<Setosa>, <Versicolor>, <Virginica>]
    # 0 : True positive
    # 1 : False Positive
    # 2 : False negative
    # 3 : true negative
    list_result = ['Setosa', 'Versicolor', 'Virginica']
    cm = [[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i,row in oldData.iterrows():
        true_result = row[target_attribute]
        # print(true_result +' -> '+result[i])
        if true_result == 'Setosa':
            if result[i] == 'Setosa':
                cm[0][0] += 1
                cm[1][3] += 1
                cm[2][3] += 1
            elif result[i] == 'Versicolor':
                cm[0][2] += 1
                cm[1][1] += 1
                cm[2][3] += 1
            elif result[i] == 'Virginica':
                cm[0][2] += 1
                cm[1][3] += 1
                cm[2][1] += 1
            else:
                print('Error when creating confusion matrix')
                break
        elif true_result == 'Versicolor':
            if result[i] == 'Setosa':
                cm[0][1] += 1
                cm[1][2] += 1
                cm[2][3] += 1
            elif result[i] == 'Versicolor':
                cm[0][3] += 1
                cm[1][0] += 1
                cm[2][3] += 1
            elif result[i] == 'Virginica':
                cm[0][3] += 1
                cm[1][2] += 1
                cm[2][1] += 1
            else:
                print('Error when creating confusion matrix')
                break
        elif true_result == 'Virginica':
            if result[i] == 'Setosa':
                cm[0][1] += 1
                cm[1][3] += 1
                cm[2][2] += 1
            elif result[i] == 'Versicolor':
                cm[0][3] += 1
                cm[1][1] += 1
                cm[2][2] += 1
            elif result[i] == 'Virginica':
                cm[0][3] += 1
                cm[1][3] += 1
                cm[2][0] += 1
            else:
                print('Error when creating confusion matrix')
                break
        else:
            print('Error when creating confusion matrix')
            break
    print()
    for i in range(len(list_result)):
        print('Confusion Matrix For Class '+list_result[i])
        print(f'True Positive : {cm[i][0]}')
        print(f'False Positive : {cm[i][1]}')
        print(f'False Negative : {cm[i][2]}')
        print(f'True Negative : {cm[i][3]}')
        print()

if __name__ == '__main__':
    url = 'iris.csv'
    data = pd.read_csv(url)
    attributes = ['sepal_length','sepal_width','petal_length','petal_width']
    target_attribute = ['variety']
    prediction_result = mini_batch_gradient(data,attributes,target_attribute)
    # print(prediction_result)
    createConfusionMatrix(prediction_result,data,target_attribute[0])
    # assume one hidden layer with 4 neuron 

# neuron layer:
# 4 -> 4 -> 3
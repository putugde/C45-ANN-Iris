import pandas as pd
from math import log2
import numpy as np
import warnings
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# convert column into dictionary with key as value of an atrribute and value as occurence of that value
def convert_to_dict(column):
	dictionary = {}
	for _,row in column.iterrows():
		val = row.values[0]
		if val:
			if(val in dictionary):
				dictionary[val] += 1
			else:
				dictionary[val] = 1
	return dictionary

def c45(df,attributes,target_attribute,main_tree):
	# check if a target_attribute has already pure
	target_dict = convert_to_dict(df[target_attribute])
	if len(target_dict) == 1:
		main_tree.add_attribute_name(list(target_dict)[0])
	elif len(attributes) == 0:
		max_label = -1
		label = ''
		for key in target_dict:
			if(target_dict[key] > max_label):
				max_label = target_dict[key]
				label = key
		main_tree.add_attribute_name(label)
	else:
		# find the highest information gain
		attr = ''
		gain_max = -1
		for attribute in attributes:
			gain_temp = gain_ratio(df,attribute, target_attribute)
			# print("gain {} : {}".format(attribute, gain_temp))
			if (gain_temp > gain_max):
				attr = attribute
				gain_max = gain_temp
		
		main_tree.add_attribute_name(attr)

		child_list = convert_to_dict(df[[attr]])
		new_attribute = attributes[:]
		new_attribute.remove(attr)
		for child_name in child_list:
			# query dataframe fata
			main_tree.add_child(child_name)
			df_hasil_query = df[df[attr] == child_name]
			c45(df_hasil_query,new_attribute,target_attribute,main_tree.get_child(child_name))


# this method return entropy for given dataframe, target attribute, and attributes
# df is a pandas dataframe, assumed to be preprocessed before
# target_attribute is a string denoting a df column name
def entropy(df, target_attribute):
    total = 0

    # count the amount 
    target_count = convert_to_dict(df[target_attribute])
    # print("Target count : {}".format(target_count))

    total_instance = sum(target_count[key] for key in target_count)
    # print("total_instance : {}".format(total_instance))

    for key in target_count:
        total -= (target_count[key]/total_instance) * log2(target_count[key]/total_instance)

    # print("Entropy : {}".format(total))
    return total

def gain(df, attribute, class_attribute):
		# class_attribute = df.keys()[1]
		# print(class_attribute)
		# cek kelas attribute
		# targets = df[class_attribute].unique()
		targets = list(convert_to_dict(df[class_attribute]))
		# return diff value in each attribute
		variables = list(convert_to_dict(df[attribute].to_frame()))
		# nilai I
		entropy_IG = 0
		# print(attribute)
		for var in variables:
			# print("----- {}".format(attribute))
			# init entropy value in each attribute
			ent = 0
			for target in targets:
				# a = pembilang, b = penyebut
				
				# print("{} == '{}' and {} == '{}'".format(attribute, var, class_attribute[0], target))
				a = len(df.query("{} == '{}' and {} == '{}'".format(attribute, var, class_attribute[0], target)))
				b = len(df.query("{} == '{}'".format(attribute, var)))
				# print("var : {}, target : {}".format(var, target))
				# print("a :{} b :{}".format(a, b))
				# pecahan
				if (a != 0):
					ent -= (a/b)*log2(a/b)
			
			# print("entropy {} = {} : {}".format(attribute, var, ent))
			
			# sigma dari (jumlahvalue di attr/jumlah instance total)*entropy attribut tersebut
			
			entropy_IG = entropy_IG + ((b/len(df[class_attribute])) * ent)
			# print("entropy IG {} {}".format(attribute ,entropy_IG))
			# print("entropy root : {}".format(entropy(df, class_attribute)))
		return entropy(df,class_attribute) - entropy_IG

def gain_ratio(df, attribute, class_attribute):
    info_gain = gain(df, attribute, class_attribute)
    info_split = entropy(df, [attribute])
    if info_gain == 0:
        return info_gain
    else:
        return info_gain/info_split

def preprocess_missing_attr(df,attribute,target_attribute):
	# print(df)
	for attr in attribute:
		i = 0
		for _ in df[[attr]].iterrows():
			# print(type(df.get_value(i,attr)))
			val = df.get_value(i,attr)
			if val != val:
				df_temp = df.query("{} == '{}' ".format(target_attribute[0],df.get_value(i,target_attribute[0])))
				mode_val = df_temp[[attr]].mode().get_value(0,attr)
				# print(mode_val)
				df.set_value(i,attr,mode_val)
			i += 1

def gains(dfs, attribute, class_attribute):
    temp0 = dfs[0].copy()
    temp0[attribute] = 'x'
    temp1 = dfs[1].copy()
    temp1[attribute] = 'y'
    temp = pd.concat([temp0, temp1])
    # g0 = gain(temp0, attribute, class_attribute)
    # print(dfs[0])
    # print(g0)
    # g1 = gain(temp1,attribute, class_attribute)
    # print(g1)
    # return (total1/total)*g0 + (total2/total)*g1
    res = gain(temp, attribute, class_attribute)
    return res


def split_data(df,is_discrete,attributes, class_attribute):
    for attr in attributes:
        if not is_discrete[attr]:
            df = df.sort_values(by=[attr])
            # print_full(df)
            length_data = df.shape[0]
            max_gains = -1*float("inf")
            idx_split = 0
            # print("attr : {} , {}".format(attr, idx_split))
            last_variety = ''
            last_attr_val = 0
            for i in range (length_data - 1):
                dfs = np.split(df,[i+1], axis=0)
                current_variety = (dfs[0].iloc[[-1]])['variety'].values[0]
                current_attr_val = (dfs[0].iloc[[-1]])[attr].values[0]
                if last_variety != current_variety and last_attr_val != current_attr_val:
                    last_variety = current_variety
                    last_attr_val = current_attr_val
                    # print(dfs[0].at[13,attr])
                    # print((dfs[0].iloc[[-1]])['variety'].values[0])
                    temp = gains(dfs, attr, class_attribute)
                    # print(temp)
                    if temp > max_gains :
                        idx_split = i
                        max_gains = temp
            # print("idx_split maks adalah {} untuk attr {} ".format(idx_split, attr))
            dfs = np.split(df,[idx_split], axis=0)
            # print_full(dfs[0])
            val = (dfs[0].iloc[[-1]])[attr].values[0]
            val2 = (dfs[1].iloc[[0]])[attr].values[0]
            treshold = round((val + val2) / 2, 3)
            # print("values : {}".format(val))
            # print("values : {}".format(val2))
            # print("treshold : {}".format(treshold))

            temp0 = dfs[0].copy()
            temp0[attr] = ' <= ' + str(treshold)
            temp1 = dfs[1].copy()
            temp1[attr] = ' > ' + str(treshold)
            df = pd.concat([temp0, temp1])

    return df

def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def train_test(df):
    return train_test_split(df, test_size=0.2)

def predict(df, main_tree):
    res = []
    for _,row in df.iterrows():
        res.append(main_tree.predictData(row))
    return res

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

class Tree:
    def __init__(self):
        self.childs = {}
        self.attribute_name = ''

    def add_child(self,label):
        child = Tree()
        self.childs[label] = child

    def get_child(self,label):
        return self.childs[label]

    def add_attribute_name(self,attribute_name):
        self.attribute_name = attribute_name

    def get_attribute_name(self):
        return self.attribute_name

    def is_leaf(self):
        return len(self.childs)==0

    def prune_tree(self):
        if self.is_leaf():
            return self.get_attribute_name()
        elif len(self.childs)==1:
            res = None
            for child in self.childs:
                res = self.childs[child].prune_tree()
            if res:
                self.childs.clear()
                self.add_attribute_name(res)
            return res
        else:
            for child in self.childs:
                self.childs[child].prune_tree()
            return ''

    def printTree(self,attribute_name,label_name,indent,is_discrete):
        for _ in range(indent):
            print('|   ',end='')
        if(attribute_name != ''):
            print('|---',end='')
            if self.is_leaf():
                if not is_discrete[attribute_name]:
                    print(f'{attribute_name}{label_name}')
                else:
                    print(f'{attribute_name} == {label_name}')
                for _ in range(indent+1):
                    print('|   ',end='')
                print('|---',end='')
                print(f'class: {self.attribute_name}')
            else:
                if not is_discrete[attribute_name]:
                    print(f'{attribute_name}{label_name}')
                else:
                    print(f'{attribute_name} == {label_name}')
                for child_name in self.childs:
                    self.childs[child_name].printTree(self.attribute_name,child_name,indent+1,is_discrete)
        else:
            for child_name in self.childs:
                    self.childs[child_name].printTree(self.attribute_name,child_name,indent,is_discrete)
    
    def predictData(self,row):
        if self.is_leaf():
            return self.attribute_name
        else:
            child_names = list(self.childs.keys())
            n = child_names[0].strip().split()
            n1 = child_names[1].strip().split()
            div_value = float(n[1])
            if float(row[self.attribute_name]) <= div_value:
                if (n[0] == '<='):
                    return self.childs[child_names[0]].predictData(row)
                elif (n1[0] == '<='):
                    return self.childs[child_names[1]].predictData(row)
                else:
                    return 'error'
            else:
                if (n[0] == '>'):
                    return self.childs[child_names[0]].predictData(row)
                elif(n1[0] == '>'):
                    return self.childs[child_names[1]].predictData(row)
                else:
                    return 'error'
            



if __name__ == '__main__':
    url = 'iris.csv'
    data = pd.read_csv(url)
    old_data = pd.read_csv(url)
    attributes = ['sepal_length','sepal_width','petal_length','petal_width']
    target_attribute = ['variety']
    is_discrete = {
        'sepal_length':False,
        'sepal_width':False,
        'petal_length':False,
        'petal_width':False,
    }
    # train,test = train_test(data)
    preprocess_missing_attr(data,attributes,target_attribute)
    preprocess_missing_attr(old_data,attributes,target_attribute)
    data = split_data(data, is_discrete, attributes, target_attribute)
    main_tree = Tree()
    c45(data,attributes,target_attribute,main_tree)
    main_tree.prune_tree()
    main_tree.printTree('','',0,is_discrete)

    # Below is the result of the prediction model
    result = predict(old_data, main_tree)

    # Create Confusion Matrix
    createConfusionMatrix(result, old_data, target_attribute[0])
    
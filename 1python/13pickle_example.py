import pickle

a_dict = {'da': 111, 2: [23,1,4], '23': {1:2,'d':'sad'}}

# pickle a variable to a file   save
file = open('pickle_example.pickle', 'wb')
pickle.dump(a_dict, file)
file.close()

# reload a file to a variable
with open('pickle_example.pickle', 'rb') as file:
    a_dict1 =pickle.load(file)
print(a_dict1)

#pickle 存的不能用记事本打开, 应该会更节约存储空间, 
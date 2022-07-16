# Core libraries
import csv
import pandas as pd


"""
vote for each class
"""
file = '/home/io18230/Desktop/label.csv'
reader = csv.reader(open(file, 'r'), delimiter=',')#
reader = list(reader)
del reader[0]


correct_pre = []

pre_label = [row[2]for row in reader]
folder = [row[3]for row in reader]
name = [row[1][:3]for row in reader]

piece = []

for i in range(len(name)-1):
    if name[i+1] != name[i]:
        piece.append(i)
piece.insert(0,-1)
piece.append(len(pre_label)-1)


for i in range(len(piece)-1):
    m1 = piece[i]
    m2 = piece[i+1]
    sub_piece = pre_label[m1+1:m2+1]

    maxlabel = max(sub_piece, key=sub_piece.count)
    if sub_piece.count(maxlabel) != len(sub_piece):
        tem = [maxlabel for n in range(0,len(sub_piece))]
    else:
        tem = sub_piece

    correct_pre = correct_pre + tem

df = pd.read_csv(file)
df['xx'] = pd.DataFrame(correct_pre)
df.to_csv(file, index=None)




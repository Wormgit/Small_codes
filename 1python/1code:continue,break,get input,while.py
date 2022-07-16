import random

# # guess 大小
number = int(random.uniform(1,10))
#number = 23

while 1:
	input_ = input('Enter an integer : ')
	if input_ == 'quit':
		break
	else:
		guess = int(input_)    #int
		if guess == number:
			print('Congratulations, you guessed it.')
			break
		elif guess < number:
			print('No, it is a little higher than that')
		else:
			print('No, it is a little lower than that')
print('Done')



print()
# very basic
condition = 0
while condition < 4:
    print(condition, end=',')
    condition = condition + 1
    
a = range(5)
while a:
    print(a[-1])
    a = a[:len(a)-1]
    
for i in range(1, 9+1,1):
    print(i)


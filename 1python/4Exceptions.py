# try + except     错误按键
# raise            #满足条件跳转


# open a txt, if it doesnot exist, create one
try:
    file=open('ee.txt','r+')
except Exception as e:
    print(e)
    response = input('do you want to create a new file:')
    if response=='y':
        file=open('eeee.txt','w')
    else:
        pass
else:
    file.write('ssss')
    file.close()




#Handling Exceptions
def Handling_Exceptions():
    try:
        text = input('Enter something --> ')
    except EOFError:   #there has to be at least one except clause associated with every try clause
        print('Why did you do an EOF on me (clt+d)?')
    except KeyboardInterrupt:
        print('You cancelled the operation.')  # clt+c
    else:
        print('You entered {}'.format(text))  # normal condition



#Raising Exceptions
class ShortInputException(Exception):
    '''A user-defined exception class.'''
    def __init__(self, length, atleast):
        Exception.__init__(self)
        self.length = length
        self.atleast = atleast

def Raising_Exceptions():
    try:
        text = input('Enter something --> ')
        if len(text) < 3:
            raise ShortInputException(len(text), 3)
        # Other work can continue as usual here
    except EOFError:
        print('Why did you do an EOF on me (clt+d)?')
    except ShortInputException as ex:
        print(('ShortInputException: The input was ' +
               '{0} long, expected at least {1}')
              .format(ex.length, ex.atleast))
    else:
        print('No exception was raised.')



## close a file correctely
#如何确保文件对象被正确关闭,无论是否会发生异常?
def close_correct():
    import sys
    import time
    f = None
    try:
        f = open("poem.txt")
        # Our usual file-reading idiom
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            print(line, end='')
            sys.stdout.flush()   #以便它能被立即打印到屏幕上
            print("Press ctrl+c now")
            # To make sure it runs for a while
            time.sleep(2)
    except IOError:                                    # no file
        print("Could not find file poem.txt")
    except KeyboardInterrupt:
        print("!! You cancelled the reading from the file.")
    finally:
        if f:
            f.close()
        print("(Cleaning up: Closed the file)")


if __name__ == '__main__':
    #Handling_Exceptions()
    #Raising_Exceptions()

    #close_correct()
    #干净的姿态得以完成close_correct
    with open("poem.txt") as f:
        for line in f:
            print(line, end='')

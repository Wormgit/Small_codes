text='This is my first test.\nThis is the second line.\nThis the third line'
text=['a','b']

print(text)
my_file=open('file_txt.txt','w')   #('文件名','形式'), 'w':write;'r':read.
my_file.write(text)               #delete original and write

my_file.close()                   

text='\tThis is my first test.\n\tThis is the second line.\n\tThis is the third line'
print(text)  #使用 \t 对齐


file= open('file_txt.txt','r') 
content=file.readlines() # python_list 形式 all lines
print(content)

for item in content:
    print(item)
    
content=file.readline()  # 读取第一行
print(content)
second_read_time=file.readline()  # 读取第二行
print(second_read_time)
print(content,second_read_time)
file.close()


append_text='\nThis is appended file.'  # 为这行文字提前空行 "\n"
my_file=open('file_txt.txt','a')   # 'a'=append 以增加内容的形式打开
my_file.write(append_text)
my_file.close()

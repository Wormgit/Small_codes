import cv2

file_path = '/home/io18230/0Projects/keras-retinanet-master/path/ID/video/2020-03-11_12-34-25/RGB.avi'
cap = cv2.VideoCapture(file_path)
# file_path是文件的绝对路径，防止路径中含有中文时报错，需要解码
if cap.isOpened():  # 当成功打开视频时cap.isOpened()返回True,否则返回False
	# get方法参数按顺序对应下表（从0开始编号)
	rate = cap.get(5)   # 帧速率
	FrameNumber = cap.get(7)  # 视频文件的帧数
	duration = FrameNumber/rate  # 帧速率/视频总帧数 是时间，s
print(rate,FrameNumber,duration)
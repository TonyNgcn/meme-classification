import os

imgdir = './ok'
for filedir in os.listdir(imgdir):
	for i, filename in enumerate(os.listdir(imgdir + '/' + filedir)):
		if (filename[-4:] == '.jpg'):
			try:
				os.rename(imgdir + '/' + filedir + '/' + filename,
				          imgdir + '/' + filedir + '/' + filedir[:2] + '_' + str(i) + '.jpg')
			# os.remove(imgdir + '/' + filedir + '/' + filename)  # 删除源文件
			except PermissionError:
				print('PermissionError' + imgdir + '/' + filedir + '/' + filename)
				pass

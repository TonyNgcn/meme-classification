from PIL import Image
import os

imgdir = './img'
for a, filedir in enumerate(os.listdir(imgdir)):
	for i, filename in enumerate(os.listdir(imgdir + '/' + filedir)):
		if (filename[-4:] == '.png')or (filename[-4:] == '.jpg') or (filename[-4:] == 'jpeg'):
			print('第' + str(i) + '个文件' + str(filename))
			gifimg = Image.open(imgdir + '/' + filedir + '/' + filename)

			if (gifimg.mode != "RGB"):
				gifimg = gifimg.convert('RGB')

			gifimg.save(imgdir + '/' + filedir + '/' + str(i) + '.jpg')

			try:
				os.remove(imgdir + '/' + filedir + '/' + filename)  # 删除源文件
			except PermissionError:
				print('不能删除'+imgdir + '/' + filedir + '/' + filename)
				pass

from PIL import Image
import os

imgdir = './ok'
for filedir in os.listdir(imgdir):
	for i, filename in enumerate(os.listdir(imgdir + '/' + filedir)):
		if (filename[-4:] == '.jpg'):
			print('第' + str(i) + '个文件' + str(filename))
			gifimg = Image.open(imgdir + '/' + filedir + '/' + filename)
			for a in range(3):
				gifimg = gifimg.rotate(90)
				if (gifimg.mode != "RGB"):
					gifimg = gifimg.convert('RGB')

				gifimg.save(imgdir + '/' + filedir + '/' + filename[:-4] + '_' + str(a) + '.jpg')

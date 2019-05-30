from PIL import Image
import os

imgdir='./img'
for a,filedir in enumerate(os.listdir(imgdir)):
	for i,filename in enumerate(os.listdir(imgdir+'/'+filedir)):
		if (filename[-4:]=='.gif'):
			print('第'+str(i)+'个文件'+str(filename))
			gifimg=Image.open(imgdir+'/'+filedir+'/'+filename)
			try:
				num=0
				for k in range(3):
					num=num+k*20
					gifimg.seek(num)
					gifimg=gifimg.copy()
					if (gifimg.mode == "P"):
						gifimg = gifimg.convert('RGB')
					gifimg.save(imgdir+'/'+filedir+'/'+str(num)+filename[:-4]+'.jpg')
					print('ok'+imgdir+'/'+filedir+'/'+str(num)+filename[:-4]+'.jpg')
			except EOFError:
				print('EOFError' + imgdir + '/' + filedir + '/' + filename)
				pass
			try:
				os.remove(imgdir + '/' + filedir + '/' + filename)  # 删除源文件
			except PermissionError:
				print('不能删除' + imgdir + '/' + filedir + '/' + filename)
				pass
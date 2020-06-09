import flickrapi
import urllib.request
import os
import sys

API_KEY = "0521014d2a3d2375c8552ae8e6e564e4"
API_SECRET = "15524b2d861c31e7"

# 输入API的key和secret
flickr = flickrapi.FlickrAPI(API_KEY, API_SECRET, cache=True)
tag = "HDR"
path = "F://Pycharm Project/HDR/"

if __name__ == '__main__':
    try:
        # 爬取tags为tag的照片,这里可以根据自己的需要设置其它的参数，还可以根据text
        photos = flickr.walk(tags=tag, extras='url_c')
    except Exception as e:
        print('Error')
    count = 1
    for photo in photos:
        # 获得照片的url,设置大小为url_c(具体参数请参看FlickrAPI官方文档介绍)
        url = photo.get('url_c')
        if str(url) == "None":
            print("It's None!")
        elif count < 7987:
            print("已经下载了")
            count = count + 1
        else:
            # 有效url进行爬取保存，文件名从1开始
            urllib.request.urlretrieve(url, path + str(count).zfill(7) + "." + os.path.basename(url).split(".")[1])
            count = count + 1
            print(url)

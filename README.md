# Flask-chinese-image-caption-app

## requirement
*  Python 3.6  
*  Flask==0.11.1  
*  gensim==3.8.1  
*  graphviz==0.14.2  
*  h5py==2.8.0  
*  hnswlib==0.4.0  
*  jieba==0.39  
*  jieba3k==0.35.1  
*  Jinja2==2.8  
*  nltk==3.3  
*  numpy==1.19.4  
*  onnx==1.8.0  
*  opencv-python==3.4.6  
*  Pillow==8.0.1  
*  scikit-image==0.13.1  
*  scikit-learn==0.21.3  
*  torch==1.7.1+cu101  
*  torchvision==0.8.2+cu101  
*  tqdm==4.54.1  

## Download Data

### authority dataSet
*  [AIC-ICC 2017 Image Chinese Description Data Set](https://challenger.ai/?lan=zh)

*  [Flick8k-CN Image Chinese Description Data Set](http://lixirong.net/datasets/flickr8kcn)

### available dataSet

*  [baiduyun AIC-ICC Image Chinese Description Data Set PASSWARD:noh0](https://pan.baidu.com/s/1T2gVLgo8Q5qGeFfKLEsBYw)   
*  [baiduyun Flick8k-CN Image Chinese Description Data Set PASSWARD:s8gb](https://pan.baidu.com/s/1T2gVLgo8Q5qGeFfKLEsBYw) 


## Download Other Files

*  tools\sample\val：  
results2-8200.json  
results2-8200-Flickr8k-CN.json  
......  


*  image_retrieval\index:
index.bin
name_index.json


*  data\log_dense_box_bn:  
infos_dense_box_bn-best.pkl  
model-best.pth  


*  data\imagenet_weights：  
resnet101.pth  


*  resource：  
caption_train_annotations_20170902.json  
imagecaption_and_keywords.json  

# overall Flask-architecture

![Flask-architecture](https://github.com/cxingstar/Flask-chinese-image-caption-app/blob/master/Flask-architecture.png)

# web Results show
python app.py  
![web-results](https://github.com/cxingstar/Flask-chinese-image-caption-app/blob/master/web-results.png)

![web-results2](https://github.com/cxingstar/Flask-chinese-image-caption-app/blob/master/web-results2.png)

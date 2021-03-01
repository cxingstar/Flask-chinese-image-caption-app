# -*- coding:utf-8 -*-
# @Time       :2021/1/2 16:27
# @Author     :Xing CHEN
# @Site       :
# @File       :app.py
# @Software   :PyCharm
# @Dirction   :None


from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import cv2
from datetime import timedelta
import shutil
import os
from eval import eval_path
import json
from image_retrieval import match_image
import jieba.analyse

path = 'static/images/'
sourcepath = 'resource/test.jpg'
image_path = 'static/image_input/images'
with open('resource/imagecaption_and_keywords.json', encoding='utf-8') as f:
    captions_data = json.load(f)

def clearpath(path):
    for i in os.listdir(path):
        path_file = os.path.join(path, i)
        if os.path.isfile(path_file):
            os.remove(path_file)
        else:
            for f in os.listdir(path_file):
                path_file2 = os.path.join(path_file, f)
                if os.path.isfile(path_file2):
                    os.remove(path_file2)

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)

@app.route('/', methods=['POST', 'GET'])  # 添加路由
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST', 'GET'])  # 添加路由
def upload():
    caption_list = ['两个 穿着 运动服 的 男人 在 运动场 上 踢足球',
                    '足球场 上 有 两个 穿着 运动服 的 男人 在 踢足球',
                    '两个 穿着 运动服 的 男人 在 运动场 上 抢 足球',
                    '两个 穿着 不同 球衣 的 男人 在 运动场 上 争抢',
                    '足球场 上 有 两个 穿着 运动服 的 男人 在 争抢 足球']
    input_image_path = 'static/resource/test.jpg'
    retrieval_list_image_path = [
        'static/resource/test_1.jpg',
        'static/resource/test_2.jpg',
        'static/resource/test_3.jpg',
    ]
    sample_retrieval_list = ['足球场上两个身穿球服的男人在踢足球',
                      '两个身穿运动服的男人在踢足球',
                      '两个穿着运动服的男人在争抢足球场']
    sample_keywords = [
        ['足球场', 1.87, 97.02],
        ['运动服', 1.70, 97.01],
        ['运动服', 1.99, 96.92],
    ]
    if request.method == 'GET' :
        return render_template('./upload.html',
                               input_image_path = input_image_path,
                               retrieval_list_image_path = retrieval_list_image_path,
                               caption_list = caption_list,
                               retrieval_list = sample_retrieval_list,
                               keywords = sample_keywords)
    if request.method == 'POST':
        clearpath(image_path)
        f = request.files['file']
        if not (f and allowed_file(f.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
        user_input = request.form.get("name")
        print(user_input)
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        upload_path = os.path.join(basepath, 'static/images', secure_filename(f.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        print(upload_path)
        f.save(upload_path)
        img = cv2.imread(upload_path)
        cv2.imwrite(os.path.join(basepath, 'static/images', 'test.jpg'), img)
        cv2.imwrite(os.path.join(basepath, 'static/image_input/images', 'test.jpg'), img)
        eval_path.eval_path()
        with open(os.path.join(basepath, 'static/image_input/results/results.json')) as f:
            data = json.load(f)
        caption_list = data[0]['caption_list']
        caption_list = caption_list[0].split('\n')
        results_path,results_similarity,cost_time= match_image.retrieval('static/image_input/images/test.jpg')
        upload_image_path = 'static/image_input/images/test.jpg'
        upload_image_retrieval_list = ['static/match_images/test_1.jpg',
                                       'static/match_images/test_2.jpg',
                                       'static/match_images/test_3.jpg']
        retrieval_list_image_name = [each.split('\\')[-1] for each in results_path]
        test_captions_retrieval_list = [captions_data[each][0] for each in retrieval_list_image_name]
        keywords = []
        for i in range(len(test_captions_retrieval_list)):
            res = jieba.analyse.extract_tags(test_captions_retrieval_list[i],topK=1,withWeight=True)[0]
            key = res[0]
            weight = round(res[1], 2)
            cossimility = results_similarity[i]
            keywords.append([key,weight,cossimility])
        shutil.copyfile(results_path[0],os.path.join(basepath, upload_image_retrieval_list[0]))
        shutil.copyfile(results_path[1],os.path.join(basepath, upload_image_retrieval_list[1]))
        shutil.copyfile(results_path[2],os.path.join(basepath, upload_image_retrieval_list[2]))

        return render_template('./upload.html',
                               input_image_path = upload_image_path,
                               retrieval_list_image_path = upload_image_retrieval_list,
                               caption_list=caption_list,
                               retrieval_list = test_captions_retrieval_list,
                               keywords = keywords)

    return render_template('./upload.html',
                           input_image_path=input_image_path,
                           retrieval_list_image_path=retrieval_list_image_path,
                           caption_list=caption_list,
                           retrieval_list=sample_retrieval_list,
                           keywords=sample_keywords)

if __name__ == '__main__':
    # app.debug = True
    app.run(port=8899)
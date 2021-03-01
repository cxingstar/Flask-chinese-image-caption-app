from image_retrieval.utils import *
import time


# 索引测试
image_dir = r'D:\Data\caption\ai_challenger\ai_challenger_caption_train_20170902\caption_train_images_20170902' # 图片仓库的文件夹

def first_create_index():
    #第一次使用或新加入图片，使用这一段代码构建索引
    remove_nonexists()
    exists_index = index_target_dir(image_dir)                      # 对目标文件夹包含图片做递归路径索引
    update_ir_index(exists_index)                                   # 按路径索引顺序建立内容检索索引


# 检索测试
def retrieval(image_path):
    results_path = []
    results_similarity = []
    exists_index = get_exists_index()                               # 获取目标文件夹的路径索引记录
    start_time = time.time()
    results = checkout(image_path, exists_index, 3)      # 被检索图片路径，图片仓库的索引，返回结果的数量
    print('Input: '+image_path)
    for result in results:
        print(f'Similarity: {result[0]:.2f} % Matched: {result[1]}')
        results_path.append(result[1])
        results_similarity.append(f'{result[0]:.2f}')
    cost_time = f'{time.time()-start_time:.4f}'
    print(f'Match cost: {time.time()-start_time:.4f}s')


    return results_path,results_similarity,cost_time

if __name__ == '__main__':

    retrieval('target_img/test.jpg')

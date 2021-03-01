import os
import json
from tqdm import tqdm
from image_retrieval.efficient_ir import EfficientIR


ir_engine = EfficientIR()
name_index_path = 'image_retrieval/index/name_index.json' # 文件路径索引的位置


def get_file_list(target_dir):
    accepted_exts = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif', '.webp']
    file_path_list = []
    for root, dirs, files in os.walk(target_dir):
        for name in files:
            if name.lower().endswith(tuple(accepted_exts)):
                file_path_list.append(os.path.join(root, name))
    return file_path_list


def get_exists_index():
    return json.loads(open(name_index_path, 'rb').read())


def index_target_dir(target_dir):
    exists_index = []
    if os.path.exists(name_index_path):
        exists_index = json.loads(open(name_index_path, 'rb').read())
    this_index = get_file_list(target_dir)
    for i in this_index:
        if not i in exists_index:
            exists_index.append(i)
    with open(name_index_path, 'wb') as wp:
        wp.write(json.dumps(exists_index,ensure_ascii=False).encode('UTF-8'))
    return exists_index


def update_ir_index(exists_index):
    count = ir_engine.hnsw_index.get_current_count()
    for idx in tqdm(range(count, len(exists_index)), ascii=True):
        fv = ir_engine.get_fv(exists_index[idx])
        if fv is None:
            continue
        ir_engine.add_fv(fv, idx)
    ir_engine.save_index()


def remove_nonexists():
    exists_index = []
    if os.path.exists(name_index_path):
        exists_index = json.loads(open(name_index_path, 'rb').read())
    for idx in tqdm(range(len(exists_index)), ascii=True):
        if not os.path.exists(exists_index[idx]):
            exists_index[idx] = 'NOTEXISTS'
            ir_engine.hnsw_index.mark_deleted(idx)
    with open(name_index_path, 'wb') as wp:
        wp.write(json.dumps(exists_index,ensure_ascii=False).encode('UTF-8'))


def checkout(image_path, exists_index, match_n=5):
    fv = ir_engine.get_fv(image_path)
    sim, ids = ir_engine.match(fv, match_n)
    return [(sim[i], exists_index[ids[i]]) for i in range(len(ids))]


def get_duplicate(exists_index, threshold):
    matched = set()
    for idx in tqdm(range(len(exists_index)), ascii=True):
        match_n = 5
        try:
            fv = ir_engine.hnsw_index.get_items([idx])[0]
        except RuntimeError:
            continue
        sim, ids = ir_engine.match(fv, match_n)
        while sim[-1] > threshold:
            match_n = round(match_n*1.5)
            sim, ids = ir_engine.match(fv, match_n)
        for i in range(len(ids)):
            if ids[i] == idx:
                continue
            if sim[i] < threshold:
                continue
            if ids[i] in matched:
                continue
            if not idx in matched:
                matched.add(idx)
            yield (exists_index[idx], exists_index[ids[i]], sim[i])

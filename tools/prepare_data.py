import os, cv2, json, shutil
import codecs


def get_list_file_in_folder(dir, ext=['jpg', 'png', 'JPG', 'PNG']):
    included_extensions = ext
    file_names = [fn for fn in os.listdir(dir)
                  if any(fn.endswith(ext) for ext in included_extensions)]
    return file_names


def convert_PICK_data_to_mmocr(PICK_data_dir, PICK_key_file, entity_list, output_mmocr_data_dir):
    boxes_and_transcripts_dir = os.path.join(PICK_data_dir, 'boxes_and_transcripts')
    images_dir = os.path.join(PICK_data_dir, 'images')
    list_img_files = get_list_file_in_folder(images_dir)
    list_img_files = sorted(list_img_files)

    print('create dict.txt')

    with open(PICK_key_file, 'r', encoding='utf-8') as f:
        classes_str = f.readlines()
    classes = [ch for ch in classes_str[0] if ch !=' ']
    dict_str = '\n'.join(classes)
    dict_str=dict_str.rstrip('\n')

    with open(os.path.join(output_mmocr_data_dir, 'dict.txt'), 'w') as f:
        f.write(dict_str)

    print('create class_list.txt')
    class_list_txt = '0 ignore\n'
    for idx, ent in enumerate(entity_list):
        line = '{} {}'.format(idx + 1, ent)
        class_list_txt += line + '\n'
    class_list_txt = class_list_txt.rstrip('\n')
    with open(os.path.join(output_mmocr_data_dir, 'class_list.txt'), 'w') as f:
        f.write(class_list_txt)

    print('get train list, val list and test list from PICK dataset')
    PICK_train_list = []
    with open(os.path.join(PICK_data_dir, 'train_list.csv'), 'r', encoding='utf-8') as f:
        train_list = f.readlines()
    for line in train_list:
        idx = -1
        for i in range(0, 2):
            idx = line.find(',', idx + 1)
        img_name = line[idx + 1:].replace('\n', '')
        PICK_train_list.append(img_name)

    PICK_val_list = []
    with open(os.path.join(PICK_data_dir, 'val_list.csv'), 'r', encoding='utf-8') as f:
        val_list = f.readlines()
    for line in val_list:
        idx = -1
        for i in range(0, 2):
            idx = line.find(',', idx + 1)
        img_name = line[idx + 1:].replace('\n', '')
        PICK_val_list.append(img_name)

    PICK_test_list = []
    with open(os.path.join(PICK_data_dir, 'test_list.csv'), 'r', encoding='utf-8') as f:
        test_list = f.readlines()
    for line in test_list:
        idx = -1
        for i in range(0, 2):
            idx = line.find(',', idx + 1)
        img_name = line[idx + 1:].replace('\n', '')
        PICK_test_list.append(img_name)

    print('convert annotation from PICK to mmocr')
    train_mmocr_txt = ''
    val_mmocr_txt = ''
    test_mmocr_txt = ''
    entity_not_in_entity_list = []
    for idx, img_name in enumerate(list_img_files):
        print(idx, img_name)
        img = cv2.imread(os.path.join(images_dir, img_name))
        mmocr_dict = {'file_name': 'image_files/{}'.format(img_name),
                      'height': img.shape[1],
                      'width': img.shape[0],
                      'annotations': []}
        tsv_path = os.path.join(boxes_and_transcripts_dir, img_name.split('.')[0] + '.tsv')
        if not os.path.exists(tsv_path):
            print('file not exist', tsv_path)
            continue
        with open(tsv_path, mode='r', encoding='utf-8') as f:
            tsv_annos = f.readlines()
        list_mmocr_annotations = []
        for anno in tsv_annos:
            mmocr_anno = {'box': [], 'text': '', 'label': 30}
            idx = -1
            for i in range(0, 9):
                idx = anno.find(',', idx + 1)
            first_comma_idx = anno.find(',')
            coordinates = anno[first_comma_idx + 1:idx]
            box = [float(f) for f in coordinates.split(',')]
            val = anno[idx + 1:].replace('\n', '')
            last_comma_idx = val.rfind(',')
            type_str = val[last_comma_idx + 1:]
            val = val[:last_comma_idx]
            if type_str not in entity_list:
                if type_str not in entity_not_in_entity_list:
                    entity_not_in_entity_list.append(type_str)
                type_str = 'other'
            label = entity_list.index(type_str) + 1
            mmocr_anno['box'] = box
            mmocr_anno['text'] = val.replace(' ', '')
            mmocr_anno['label'] = label
            list_mmocr_annotations.append(mmocr_anno)
        mmocr_dict['annotations'] = list_mmocr_annotations
        mmocr_dict_str = json.dumps(mmocr_dict, ensure_ascii=False)
        if img_name in PICK_train_list:
            train_mmocr_txt += mmocr_dict_str + '\n'
        elif img_name in PICK_val_list:
            val_mmocr_txt += mmocr_dict_str + '\n'
        elif img_name in PICK_test_list:
            test_mmocr_txt += mmocr_dict_str + '\n'
        else:
            print(img_name, 'not in PICK train list, PICK val list or PICK test list')
    train_mmocr_txt= train_mmocr_txt.rstrip('\n')
    val_mmocr_txt= val_mmocr_txt.rstrip('\n')
    test_mmocr_txt= test_mmocr_txt.rstrip('\n')

    print(entity_not_in_entity_list, 'not in entity list')

    with open(os.path.join(output_mmocr_data_dir, 'train.txt'), 'w', encoding='utf-8') as outfile:
        outfile.write(train_mmocr_txt)
    with open(os.path.join(output_mmocr_data_dir, 'val.txt'), 'w', encoding='utf-8') as outfile:
        outfile.write(val_mmocr_txt)
    with open(os.path.join(output_mmocr_data_dir, 'test.txt'), 'w', encoding='utf-8') as outfile:
        outfile.write(test_mmocr_txt)

    print('copy imgs file from PICK to mmocr')
    mmocr_img_dir = os.path.join(output_mmocr_data_dir, 'image_files')
    if not os.path.exists(mmocr_img_dir):
        os.makedirs(mmocr_img_dir)

    for idx, img_name in enumerate(list_img_files):
        if os.path.exists(os.path.join(mmocr_img_dir, img_name)):
            continue
        print(idx, img_name)
        shutil.copy(os.path.join(images_dir, img_name),
                    os.path.join(mmocr_img_dir, img_name))
    print('Done!')

if __name__ == '__main__':
    PICK_data_dir = '/home/duycuong/home_data/mmocr/kie/sale_contracts_PICK'
    PICK_key_file = '/home/duycuong/PycharmProjects/vvn/demo_read_document/demo_app/kie_models/PICK/utils/keys_vietnamese.txt'
    output_mmocr_data_dir = '/home/duycuong/home_data/mmocr/kie/sale_contracts'
    Entities_list = ['contract_no','exporter_name','exporter_add','importer_name','payment_method',
                 'ben_bank_name','ben_add','ben_name','ben_acc','swift_code', 'other']

    convert_PICK_data_to_mmocr(PICK_data_dir=PICK_data_dir,
                               PICK_key_file=PICK_key_file,
                               entity_list=Entities_list,
                               output_mmocr_data_dir=output_mmocr_data_dir)

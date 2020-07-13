import os
import xmlrpc.client
import json
import shutil
import cv2
import my_config

from my_module import my_preprocess

def get_disease_name(no, class_type='Stage', lang='en'):
    if class_type == 'Gradable':
        json_file = 'classid_to_human_Gradable.json'
    elif class_type == 'Left_Right':
        json_file = 'classid_to_human_Left_Right.json'
    elif class_type == 'Stage':
        json_file = 'classid_to_human_Stage.json'
    elif class_type == 'Hemorrhage':
        json_file = 'classid_to_human_Hemorrhage.json'
    elif class_type == 'Posterior':
        json_file = 'classid_to_human_Posterior.json'
    elif class_type == 'Plus':
        json_file = 'classid_to_human_Plus.json'
    else:
        raise Exception('get disease name error!')

    if lang == 'cn':
        json_file = 'cn_' + json_file

    baseDir = os.path.dirname(os.path.abspath(__name__))
    json_file = os.path.join(os.path.join(baseDir, 'diseases_json'), json_file)

    with open(json_file, 'r') as json_file:
        data = json.load(json_file)
        for i in range(len(data['diseases'])):
            if data['diseases'][i]['NO'] == no:
                return data['diseases'][i]['NAME']

def predict_single_class(img_source, class_type='Stage', softmax_or_multilabels='softmax'):

    if class_type == 'Gradable':  # image quality
        SERVER_URL = my_config.URL_GRADABLE
    elif class_type == 'Left_Right':
        SERVER_URL = my_config.URL_LEFT_RIGHT

    elif class_type == 'Stage':
        SERVER_URL = my_config.URL_STAGE
    elif class_type == 'Hemorrhage':  # HEMORRHAGE
        SERVER_URL = my_config.URL_HEMORRHAGE
    elif class_type == 'Posterior':
        SERVER_URL = my_config.URL_POSTERIOR
    # elif class_type == 'Plus':  # Plus one stage
    #     SERVER_URL = my_config.URL_PLUS_ONE_STAGE
    elif class_type == 'Plus':  # Plus two stages
        SERVER_URL = my_config.URL_PLUS_TWO_STAGES

    with xmlrpc.client.ServerProxy(SERVER_URL) as proxy1:
        if softmax_or_multilabels == 'softmax':
            prob_list, pred_list, prob_total, pred_total, correct_model_no = proxy1.predict_softmax(
                img_source)
        else:
            prob_list, pred_list, prob_total, pred_total, correct_model_no = proxy1.predict_multi_labels(
                img_source)
        return prob_list, pred_list, prob_total, pred_total, correct_model_no


def predict_blood_vessel_seg(img_source, preprocess=True):
    with xmlrpc.client.ServerProxy(my_config.URL_BLOOD_VESSEL_SEG) as proxy1:
        filename_seg = proxy1.server_seg_blood_vessel(img_source, preprocess)
        return filename_seg

def predict_optic_disc_seg(img_file_source, preprocess=True, img_file_blood_seg=None):
    with xmlrpc.client.ServerProxy(my_config.URL_DETECT_OPTIC_DISC) as proxy1:
        confidence, img_file_mask_tmp, img_file_crop_optic_disc, img_file_draw_circle, img_file_blood_vessel_seg_posterior =\
            proxy1.detect_optic_disc(img_file_source, preprocess, img_file_blood_seg)
        return confidence, img_file_mask_tmp, img_file_crop_optic_disc, img_file_draw_circle, img_file_blood_vessel_seg_posterior


def predict_all(file_img_source, str_uuid, baseDir, lang,
                show_deepshap_stage=False,
                show_deepshap_hemorrhage=False, show_deepshap_plus=False):

    img_source = cv2.imread(file_img_source)
    predict_result = {}

    #region preprocess
    img_file_resized_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'resized_384.jpg')
    cv2.imwrite(img_file_resized_384, cv2.resize(img_source, (384, 384)))
    predict_result['img_file_resized_384'] = img_file_resized_384.replace(baseDir, '')

    img_file_preprocessed_384 = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'preprocessed_384.jpg')
    my_preprocess.do_preprocess(img_source,  crop_size=384,
                img_file_dest=img_file_preprocessed_384, add_black_pixel_ratio=0.02)
    predict_result['img_file_preprocessed_384'] = img_file_preprocessed_384.replace(baseDir, '')
    #endregion

    prob_list_gradable, pred_list_gradable, prob_total_gradable, pred_total_gradable, correct_model_no_gradable = \
            predict_single_class(img_file_preprocessed_384, class_type='Gradable', softmax_or_multilabels='softmax')
    predict_result['img_gradable'] = pred_total_gradable
    predict_result["img_gradable_0_name"] = get_disease_name(0, class_type='Gradable', lang=lang)
    predict_result["img_gradable_0_prob"] = round(prob_total_gradable[0], 3) * 100
    predict_result["img_gradable_1_name"] = get_disease_name(0, class_type='Gradable', lang=lang)
    predict_result["img_gradable_1_prob"] = round(prob_total_gradable[1], 3) * 100

    if predict_result['img_gradable'] == 1: #image quality ungradable
        if my_config.UNGRADABLE_RESHOOTING:
            if lang == 'en':
                predict_result['total_results'] = 'Rephotograph'
            else:
                predict_result['total_results'] = '诊断结果是:图片质量不可评级，建议重拍。'

            predict_result['recommended'] = predict_result['total_results']

            return predict_result

    if my_config.ENABLE_LEFT_RIGHT:
        prob_list_left_right, pred_list_left_right, prob_total_left_right, pred_total_left_right, correct_model_no_left_right = \
                predict_single_class(img_file_preprocessed_384, class_type='Left_Right', softmax_or_multilabels='softmax')
        predict_result['img_left_right'] = pred_total_left_right
        predict_result["img_left_right_0_name"] = get_disease_name(0, class_type='Left_Right', lang=lang)
        predict_result["img_left_right_0_prob"] = round(prob_total_left_right[0], 3) * 100
        predict_result["img_left_right_1_name"] = get_disease_name(1, class_type='Left_Right', lang=lang)
        predict_result["img_left_right_1_prob"] = round(prob_total_left_right[1], 3) * 100


    prob_list_stage, pred_list_stage, prob_total_stage, pred_total_stage, correct_model_no_stage = \
        predict_single_class(img_file_preprocessed_384, class_type='Stage', softmax_or_multilabels='softmax')
    predict_result['img_stage'] = pred_total_stage
    predict_result["img_stage_0_name"] = get_disease_name(0, class_type='Stage', lang=lang)
    predict_result["img_stage_0_prob"] = round(prob_total_stage[0], 3) * 100
    predict_result["img_stage_1_name"] = get_disease_name(1, class_type='Stage', lang=lang)
    predict_result["img_stage_1_prob"] = round(prob_total_stage[1], 3) * 100

    prob_list_hemorrhage, pred_list_hemorrhage, prob_total_hemorrhage, pred_total_hemorrhage, correct_model_no_hemorrhage = \
        predict_single_class(img_file_preprocessed_384, class_type='Hemorrhage', softmax_or_multilabels='softmax')
    predict_result['img_hemorrhage'] = pred_total_hemorrhage
    predict_result["img_hemorrhage_0_name"] = get_disease_name(0, class_type='Hemorrhage', lang=lang)
    predict_result["img_hemorrhage_0_prob"] = round(prob_total_hemorrhage[0], 2) * 100
    predict_result["img_hemorrhage_1_name"] = get_disease_name(1, class_type='Hemorrhage', lang=lang)
    predict_result["img_hemorrhage_1_prob"] = round(prob_total_hemorrhage[1], 2) * 100

    prob_list_posterior, pred_list_posterior, prob_total_posterior, pred_total_posterior, correct_model_no_posterior = \
        predict_single_class(img_file_preprocessed_384, class_type='Posterior', softmax_or_multilabels='softmax')
    predict_result['img_posterior'] = pred_total_posterior
    predict_result["img_posterior_0_name"] = get_disease_name(0, class_type='Posterior', lang=lang)
    predict_result["img_posterior_0_prob"] = round(prob_total_posterior[0], 2) * 100
    predict_result["img_posterior_1_name"] = get_disease_name(1, class_type='Posterior', lang=lang)
    predict_result["img_posterior_1_prob"] = round(prob_total_posterior[1], 2) * 100

    if predict_result['img_posterior'] == 0:  #is posterior
        tmp_file_blood_vessel_seg = predict_blood_vessel_seg(file_img_source, preprocess=True)
        filename_blood_vessel_seg = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'blood_vessel_seg.jpg')
        shutil.copy(tmp_file_blood_vessel_seg, filename_blood_vessel_seg)
        predict_result["blood_vessel_seg_file"] = filename_blood_vessel_seg.replace(baseDir, '')

        #optic disc seg
        confidence, img_file_mask_tmp, img_file_crop_optic_disc, \
            img_file_draw_circle, img_file_blood_vessel_seg_posterior =\
            predict_optic_disc_seg(file_img_source, True,  filename_blood_vessel_seg)
        if confidence is not None:
            if my_config.SHOW_OPTIC_DISC_MASK:
                filename_optic_disc_mask = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'optic_disc_mask.jpg')
                shutil.copy(img_file_mask_tmp, filename_optic_disc_mask)
                predict_result["filename_optic_disc_mask"] = filename_optic_disc_mask.replace(baseDir, '')

            filename_draw_circle = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'draw_circle.jpg')
            shutil.copy(img_file_draw_circle, filename_draw_circle)
            predict_result["filename_draw_circle"] = filename_draw_circle.replace(baseDir, '')

            if my_config.SHOW_BLOOD_VESSEL_SEG_POSTERIOR:
                filename_blood_vessel_seg_posterior = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'blood_vessel_seg_posterior.jpg')
                shutil.copy(img_file_blood_vessel_seg_posterior, filename_blood_vessel_seg_posterior)
                predict_result["filename_blood_vessel_seg_posterior"] = filename_blood_vessel_seg_posterior.replace(baseDir, '')

            #plus two stages classification  input blood vessel seg file
            prob_list_plus, pred_list_plus, prob_total_plus, pred_total_plus, correct_model_no_plus = \
                predict_single_class(filename_blood_vessel_seg, class_type='Plus', softmax_or_multilabels='softmax')
            predict_result['img_plus'] = pred_total_plus
            predict_result["img_plus_0_name"] = get_disease_name(0, class_type='Plus', lang=lang)
            predict_result["img_plus_0_prob"] = round(prob_total_plus[0], 2) * 100
            predict_result["img_plus_1_name"] = get_disease_name(1, class_type='Plus', lang=lang)
            predict_result["img_plus_1_prob"] = round(prob_total_plus[1], 2) * 100

    predict_result['show_deepshap_stage'] = show_deepshap_stage
    predict_result['show_deepshap_hemorrhage'] = show_deepshap_hemorrhage
    predict_result['show_deepshap_plus'] = show_deepshap_plus

    if show_deepshap_stage or show_deepshap_hemorrhage or show_deepshap_plus:
        predict_result["Heatmap_deepshap"] = 1
    else:
        predict_result["Heatmap_deepshap"] = 0

    if show_deepshap_stage and predict_result['img_stage'] == 1:
        with xmlrpc.client.ServerProxy(my_config.URL_DEEPSHAP_STAGE) as proxy1:
            model_no = 0
            list_classes, list_images = proxy1.server_shap_deep_explainer(model_no,
                    img_file_preprocessed_384, False, 1, my_config.BLEND_ORIGINAL_IMAGE)

            filename_heatmap = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'Heatmap_deepshap_stage_1.jpg')
            shutil.copy(list_images[0], filename_heatmap)
            predict_result["Heatmap_deepshap_stage"] = filename_heatmap.replace(baseDir, '')

    if show_deepshap_hemorrhage and predict_result['img_hemorrhage'] == 1:
        with xmlrpc.client.ServerProxy(my_config.URL_DEEPSHAP_HEMORRHAGE) as proxy1:
            model_no = 1
            list_classes, list_images = proxy1.server_shap_deep_explainer(model_no,
                    img_file_preprocessed_384, False, 1, my_config.BLEND_ORIGINAL_IMAGE)

            filename_heatmap = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'Heatmap_deepshap_hemorrhage_1.jpg')
            shutil.copy(list_images[0], filename_heatmap)
            predict_result["Heatmap_deepshap_hemorrhage"] = filename_heatmap.replace(baseDir, '')

    if show_deepshap_plus and predict_result['img_plus'] == 1:
        with xmlrpc.client.ServerProxy(my_config.URL_DEEPSHAP_PLUS) as proxy1:
            model_no = 2
            list_classes, list_images = proxy1.server_shap_deep_explainer(model_no,
                    img_file_preprocessed_384, False, 1, my_config.BLEND_ORIGINAL_IMAGE)

            filename_heatmap = os.path.join(baseDir, 'static', 'imgs', str_uuid, 'Heatmap_deepshap_plus_1.jpg')
            shutil.copy(list_images[0], filename_heatmap)
            predict_result["Heatmap_deepshap_plus"] = filename_heatmap.replace(baseDir, '')

    #region disease name
    if lang == 'en':
        predict_result['total_results'] = ''
    else:
        predict_result['total_results'] = '建议 : '

    if (predict_result['img_stage'] == 1 or predict_result['img_stage'] == 1) \
        or (predict_result['img_hemorrhage'] == 1 or predict_result['img_hemorrhage'] == 1):

        referal = True
        if lang == 'en':
            predict_result['total_results'] += 'Referral-warranted'
        else:
            predict_result['total_results'] += '转诊'

    else:
        if ('img_plus' in predict_result) and (predict_result['img_plus'] == 1):
            referal = True
            if lang == 'en':
                predict_result['total_results'] += 'Referral-warranted'
            else:
                predict_result['total_results'] += '转诊'
        else:
            referal = False
            if lang == 'en':
                predict_result['total_results'] += 'Follow up'
            else:
                predict_result['total_results'] += '随访'

    if predict_result['img_gradable'] == 1 :
        predict_result['total_results'] = 'Rephotograph'

    predict_result['detected'] = ''

    if referal:
        if lang == 'en':
            disease_name = ' Detected positive ROP-related feature: '
        else:
            disease_name = '检测到 : '
        if predict_result['img_stage'] == 1:
            disease_name += get_disease_name(predict_result['img_stage'],
                                             class_type='Stage', lang=lang)
            disease_name += ', '

        if predict_result['img_hemorrhage'] == 1:
            disease_name += get_disease_name(predict_result['img_hemorrhage'],
                                             class_type='Hemorrhage', lang=lang)
            disease_name += ', '

        if ('img_plus' in predict_result) and (predict_result['img_plus'] == 1):
            disease_name += get_disease_name(predict_result['img_plus'],
                                             class_type='Plus', lang=lang)
            disease_name += ', '

        disease_name = disease_name[:-2]
        disease_name += ''
        # if lang == 'en':
        #     predict_result['total_results'] += '. '
        # else:
        #     predict_result['total_results'] += '。 '

        predict_result['detected'] = disease_name

    predict_result['recommended'] = predict_result['total_results']
    predict_result['total_results'] = predict_result['detected'] + '\n' + predict_result['total_results']

    #endregion

    return predict_result


def save_to_db_diagnose(ip, username, image_uuid, diagnostic_results, del_duplicate=False):
    from my_module import db_helper
    db = db_helper.get_db_conn()
    cursor = db.cursor()

    if del_duplicate:
        if db_helper.DB_TYPE == 'mysql':
            sql = "delete from tb_diagnoses where image_uuid = %s"
        if db_helper.DB_TYPE == 'sqlite':
            sql = "delete from tb_diagnoses where image_uuid = ?"
        cursor.execute(sql, (image_uuid,))
        db.commit()

    if db_helper.DB_TYPE == 'mysql':
        sql = "insert into tb_diagnoses(IP,username, image_uuid, diagnostic_results) values(%s,%s,%s,%s) "
    if db_helper.DB_TYPE == 'sqlite':
        sql = "insert into tb_diagnoses(IP,username, image_uuid, diagnostic_results) values(?,?,?,?) "
    cursor.execute(sql, (ip, username, image_uuid, diagnostic_results))
    db.commit()

    db.close()


def reload_my_config():
    from imp import reload
    reload(my_config)
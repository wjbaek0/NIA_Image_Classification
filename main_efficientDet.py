'''
최종수정일자 : 2020-12-07
수정자 : 백우준
설명 : mAP를 계산하고, confusion matrix를 추론하여 output, matrix 폴더에 결과를 저장하는 파일 
사용법 : ~/mAP-master$ python main_final.py 입력 후, 각 모델명 입력
'''
import glob
import json
import os
import shutil
import operator
import sys
import argparse
import math
from collections import Counter
import time
import pandas as pd
import numpy as np

MINOVERLAP = 0.5 # default value (defined in the PASCAL VOC2012 challenge) IOU값

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored (e.g., python main.py --ignore person book)
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU (e.g., python main.py --set-class-iou person 0.7)
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

'''
    0,0 ------> x (width)
     |
     |  (Left,Top)
     |      *_________
     |      |         |
            |         |
     y      |_________|
  (height)            *
                (Right,Bottom)
        
'''

# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
    args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
    specific_iou_flagged = True

file = input("detecting class name ? : ")
file_name = file + ".txt"

if file == "rootcanal" or file == "prosthesis" :

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth', file_name)

    DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results', file_name)



    # if there are no images then no animation can be shown
    IMG_PATH = os.path.join(os.getcwd(), 'input', 'images-optional')
    if os.path.exists(IMG_PATH): 
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                # no image files found
                args.no_animation = True
    else:
        args.no_animation = True

    # try to import OpenCV if the user didn't choose the option --no-animation
    show_animation = False
    if not args.no_animation:
        try:
            import cv2
            show_animation = True
        except ImportError:
            print("\"opencv-python\" not found, please install to visualize the results.")
            args.no_animation = True

    # try to import Matplotlib if the user didn't choose the option --no-plot
    draw_plot = False
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            draw_plot = True
        except ImportError:
            print("\"matplotlib\" not found, please install it to get the resulting plots.")
            args.no_plot = True


    def log_average_miss_rate(prec, rec, num_images):
        """
            log-average miss rate:
                Calculated by averaging miss rates at 9 evenly spaced FPPI points
                between 10e-2 and 10e0, in log-space.

            output:
                    lamr | log-average miss rate
                    mr | miss rate
                    fppi | false positives per image

            references:
                [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
                State of the Art." Pattern Analysis and Machine Intelligence, IEEE
                Transactions on 34.4 (2012): 743 - 761.
        """

        # if there were no detections of that class
        if prec.size == 0:
            lamr = 0
            mr = 1
            fppi = 0
            return lamr, mr, fppi

        fppi = (1 - prec)
        mr = (1 - rec)

        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)

        # Use 9 evenly spaced reference points in log-space
        ref = np.logspace(-2.0, 0.0, num = 9)
        for i, ref_i in enumerate(ref):
            # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]

        # log(0) is undefined, so we use the np.maximum(1e-10, ref)
        lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

        return lamr, mr, fppi

    """
    throw error and exit
    """
    def error(msg):
        print(msg)
        sys.exit(0)

    """
    check if the number is a float between 0.0 and 1.0
    """
    def is_float_between_0_and_1(value):
        try:
            val = float(value)
            if val > 0.0 and val < 1.0:
                return True
            else:
                return False
        except ValueError:
            return False


    def voc_ap(rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """

        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        """
        This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])
        return ap, mrec, mpre


    """
    Convert the lines of a file to a list
    """
    def file_lines_to_list(path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content

    """
    Draws text in image
    """
    def draw_text_in_image(img, text, pos, color, line_width):
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        lineType = 1
        bottomLeftCornerOfText = pos
        cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
        text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
        return img, (line_width + text_width)

    """
    Plot - adjust axes
    """
    def adjust_axes(r, t, fig, axes):
        # get text width for re-scaling
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        # get axis width in inches
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        # get axis limit
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1]*propotion])

    """
    Draw plot using Matplotlib
    """
    def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
        # sort the dictionary by decreasing value, into a list of tuples
        sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(0), reverse=True)
        # sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1)) # value DESC시에 

        # unpacking the list of tuples into two lists
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)
        # 
        if true_p_bar != "":
            """
            Special case to draw in:
                - green -> TP: True Positives (object detected and matches ground-truth)
                - red -> FP: False Positives (object detected but does not match ground-truth)
                - pink -> FN: False Negatives (object not detected but present in the ground-truth) // 미구현
            """
            fp_sorted = []
            tp_sorted = []
            for key in sorted_keys:
                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])

            # TP FP 개별 추론되는 로직
            # print(fp_sorted)
            # print(tp_sorted)

            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
            # add legend
            plt.legend(loc='lower right')
            """
            Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                # trick to paint multicolor with offset:
                # first paint everything and then repaint the first number
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values)-1): # largest bar
                    adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            """
            Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val) # add a space before
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                # re-set axes to show number inside the figure
                if i == (len(sorted_values)-1): # largest bar
                    adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title(window_title)
        # write classes in y axis
        tick_font_size = 12
        plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
        """
        Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height 
        top_margin = 0.15 # in percentage of the figure height
        bottom_margin = 0.05 # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title(plot_title, fontsize=14)
        plt.xlabel(x_label, fontsize='large')
        # adjust size of window
        fig.tight_layout()
        # save the plot
        fig.savefig(output_path)
        # show image
        if to_show:
            plt.show()
        # close the plot
        plt.close()

    """
    Create a ".temp_files/" and "output/" directory
    """
    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)
    output_files_path = os.path.join("output", file)

    if os.path.exists(output_files_path): # if it exist already
        # reset the output directory
        shutil.rmtree(output_files_path)
        # print("output 폴더를 초기화 합니다.")

    os.makedirs(output_files_path)


    if draw_plot:
        os.makedirs(os.path.join(output_files_path , "classes"))
    if show_animation:
        os.makedirs(os.path.join(output_files_path , "images", "detections_one_by_one"))

    """
    ground-truth
        Load each of the ground-truth files into a temporary ".json" file.
        Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH)
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    for txt_file in ground_truth_files_list:
        # print("GT 파일 리스트 : " , txt_file)
        
        category_id = txt_file.split(".txt", 1)[0]
        category_id = os.path.basename(os.path.normpath(category_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH)
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                        class_name, image_name,  left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                else:
                        class_name, image_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            # check if class is in the ignore list, if yes skip
            if class_name in args.ignore:
                continue
            bbox = left + " " + top + " " + right + " " +bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "image_name":image_name,  "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "image_name":image_name, "bbox":bbox, "used":False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + category_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)
    print("클래스명 : " , gt_classes)
    # print(gt_counter_per_class)

    """
    Check format of the flag --set-class-iou (if used)
        e.g. check if class exists
    """
    if specific_iou_flagged:
        n_args = len(args.set_class_iou)
        error_msg = \
            '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
        if n_args % 2 != 0:
            error('Error, missing arguments. Flag usage:' + error_msg)
        # [class_1] [IoU_1] [class_2] [IoU_2]
        # specific_iou_classes = ['class_1', 'class_2']
        specific_iou_classes = args.set_class_iou[::2] # even
        # iou_list = ['IoU_1', 'IoU_2']
        iou_list = args.set_class_iou[1::2] # odd
        if len(specific_iou_classes) != len(iou_list):
            error('Error, missing arguments. Flag usage:' + error_msg)
        for tmp_class in specific_iou_classes:
            if tmp_class not in gt_classes:
                        error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
        for num in iou_list:
            if not is_float_between_0_and_1(num):
                error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

    """
    detection-results
        Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH)
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # print("detection 파일 리스트 : " , txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            category_id = txt_file.split(".txt",1)[0]
            category_id = os.path.basename(os.path.normpath(category_id))
            temp_path = os.path.join(GT_PATH)
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, image_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    #print("match")
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "image_name": image_name ,"category_id":category_id, "bbox":bbox})
                    #print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
    Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    """
    TP , FP , FN , TN  저장용 변수 선언 ##########

    """
    tp_store = 0 # TP값 저장 변수 
    fp_store = 0 # FP값 저장 변수
    fn_store = 0 # FN값 저장 변수
    tn_store = 0 # TN값 저장 변수
    pre_store = []
    recall_store = []
    F1_store = []

    # 결과를 텍스트로 저장 - /matrix/Matrix_data_ 
    dir = os.getcwd()
    if not os.path.exists(dir + '/matrix/'):
        os.makedirs(dir + '/matrix/')
    result = open(dir + '/matrix/Matrix_data_'+ file +'.txt','wt')

    text_store = ""
    # open file to store the output
    with open(output_files_path + "/output.txt", 'w') as output_file:
        output_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
            Load detection-results of that class
            """
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            """
            Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            ap_list = []
            for idx, detection in enumerate(dr_data): # 추론데이터 불러오는 시점
                category_id = detection["category_id"]
                # assign detection-results to ground truth object if any
                # open ground-truth with that category_id
                gt_file = TEMP_FILES_PATH + "/" + category_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [ float(x) for x in detection["bbox"].split() ] # 추론 bbox
                bb_name = detection["image_name"]
                for obj in ground_truth_data:            # 그라운드 데이터 불러오는 시점
                    # look for a class_name match
                    if (obj["class_name"] == class_name):
                        bbgt = [ float(x) for x in obj["bbox"].split() ] # ground bbox
                        bbgt_name = obj["image_name"] 
                        if (bb_name == bbgt_name) :
                            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:    # 기본 OVMAX 보다 높은 IOU (OV) 값이 들어올시 저장 
                                    ovmax = ov
                                    gt_match = obj # ground_truth_data 저장 

                # assign detection as true positive/don't care/false positive
                min_overlap = MINOVERLAP  # 설정한 초기 IOU값을 가져옴 
                if specific_iou_flagged:
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        min_overlap = float(iou_list[index])

                if ovmax >= min_overlap: #  추론한 IOU값이 초기설정 IOU 보다 높을 경우. 
                    if "difficult" not in gt_match:
                            if not bool(gt_match["used"]): # ground_truth_data가 "USED"로 처리되지 않은것  
                                # true positive
                                tp[idx] = 1 # TP 배열에 1 저장 
                        
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file, 'w') as f:
                                        f.write(json.dumps(ground_truth_data))
                                if show_animation:
                                    status = "MATCH!"
                            
                            else:
                                # false positive (multiple detection) / 0.5 보다 높은 FP
                                fp[idx] = 1 # FP 배열에 1 저장 
                                if show_animation:
                                    status = "REPEATED MATCH!"
                else:
                    # false positive / 0.5 보다 낮은 FP
                    fp[idx] = 1 # FP 배열에 1 저장 
                    if ovmax > 0:
                        status = "INSUFFICIENT OVERLAP"


            result.write('< '+class_name +' 치아> \n 잡힌 박스수:\t'+ str(sum(tp)+sum(fp))+'\n'+ "TP\t" + str(sum(tp)) + '개 \n' + "FP\t" + str(sum(fp))+'개 \n')
            tp_store = sum(tp)
            fp_store = sum(fp)

            for class_name in sorted(gt_counter_per_class):
                result.write("FN\t" + str(gt_counter_per_class[class_name] - tp_store)  + "개 \n" )
                fn_store = gt_counter_per_class[class_name] - tp_store
                # 임시 TN 값 
                result.write("TN\t"+ str(tn_store) + '개 \n')

            # compute precision/recall 계산 로직 #######
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(fp)
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #Recall 출력      
            #print(rec)
            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
            #Precision 출력 
            #print(prec)


            # ap를 계산하는 로직으로 보내서 값을 받아옴
            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            # our_ap = sum(prec) / len(prec)

            text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
            """
            Write to output.txt
            """
            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
            output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
            
            # pre , recall
            try :
                result.write("Precision\t" + str(rounded_prec[-1]) + '\n' + "Recall\t" + str(rounded_rec[-1]) + "\n")
                pre_store = rounded_prec[-1]
                recall_store = rounded_rec[-1]
            except :
                result.write("Precision\t0.000 " + '\n' + "Recall\t0.000 " + "\n")
                pre_store = 0.000
                recall_store = 0.000

            # F1 score
            try:
                result.write('F1-Score\t' + str(round(2.0 * (float(rounded_prec[-1]) * float(rounded_rec[-1]))/(float(rounded_prec[-1]) + float(rounded_rec[-1])) , 3))+ '\n')
            except :
                result.write('F1-Score\t0.000' + '\n')
            

            if not args.quiet:
                print(text)
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
            lamr_dictionary[class_name] = lamr

            """
            Draw plot
            """
            if draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf() # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                #plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca() # gca - get current axes
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05]) # .05 to give some extra space
                # Alternative option -> wait for button to be pressed
                #while not plt.waitforbuttonpress(): pass # wait for key display
                # Alternative option -> normal display
                #plt.show()
                # save the plot
                fig.savefig(output_files_path + "/classes/" + class_name + ".png")
                plt.cla() # clear axes for next plot


        output_file.write("\n# mAP of all classes\n")
        mAP = sum_AP / n_classes
        text = "mAP\t{0:.2f}%".format(mAP*100)
        text_store = text 
        output_file.write(text + "\n")
        print(text)

    """
    Draw false negatives // FN(Pink) 구하는 로직? 참고사항
    """
    if show_animation:
        pink = (203,192,255)
        for tmp_file in gt_files:
            ground_truth_data = json.load(open(tmp_file))
            #print(ground_truth_data)
            # get name of corresponding image
            start = TEMP_FILES_PATH + '/'
            img_id = tmp_file[tmp_file.find(start)+len(start):tmp_file.rfind('_ground_truth.json')]
            img_cumulative_path = output_files_path + "/images/" + img_id + ".jpg"
            img = cv2.imread(img_cumulative_path)
            if img is None:
                img_path = IMG_PATH + '/' + img_id + ".jpg"
                img = cv2.imread(img_path)
            # draw false negatives
            for obj in ground_truth_data:
                if not obj['used']:
                    bbgt = [ int(round(float(x))) for x in obj["bbox"].split() ]
                    cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),pink,2)
            cv2.imwrite(img_cumulative_path, img)

    # remove the temp_files directory
    # shutil.rmtree(TEMP_FILES_PATH)

    """
    Count total of detection-results
    """
    # iterate through all the files
    det_counter_per_class = {}
    for txt_file in dr_files_list:
        # get lines to list
        lines_list = file_lines_to_list(txt_file)
        for line in lines_list:
            class_name = line.split()[0]
            # check if class is in the ignore list, if yes skip
            if class_name in args.ignore:
                continue
            # count that object
            if class_name in det_counter_per_class:
                det_counter_per_class[class_name] += 1
            else:
                # if class didn't exist yet
                det_counter_per_class[class_name] = 1

    # TP + FP 총 갯수 출력.             
    # print(det_counter_per_class)
    dr_classes = list(det_counter_per_class.keys())


    """
    Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        x_label = "Number of objects per class"
        output_path = output_files_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
            )

    # mAP
    result.write(text_store + '\n')

    # 클래스 정확도
    acc = ((tp_store + tn_store ) / (tp_store + tn_store + fp_store + fn_store ))*100 # tn값을 찾아서 tn_store 값 수정 필요함!
    result.write("Accuracy\t"+ str(round(acc,2)) + ' % \n')

    
    """
    Finish counting true positives
    """
    for class_name in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0

    cm_data = {'Positive':[str(tp_store), str(fp_store), '-'],'Negative':[str(tn_store), str(fn_store), '-'], 'Precision':[str(pre_store), 'Recall', str(recall_store)],
            'F1-Score':[str(round(2.0 * (float(pre_store) * float(recall_store))/(float(pre_store) + float(recall_store)) , 3)), 'Accuracy', str(round(acc,2))], 
            'mAP':["{0:.2f}%".format(mAP*100), "All-Ground-Truth", str(tp_store + fn_store)]}

    cm_df = pd.DataFrame(cm_data, columns=['Positive', 'Negative','-', 'Precision', 'F1-Score', 'mAP'], index = ['True', 'False', '-'])


    print("\n")
    print("confusion_matrix")
    print("All-Ground-Truth = ", str(tp_store + fn_store))
    print("TP = ", str(tp_store))
    print("FP = ", str(fp_store))
    print("TN = ", str(tn_store))
    print("FN = ", str(fn_store))
    print("-------------------------------")
    print(cm_df.loc[:,"Positive":"-"])
    print("-------------------------------")
    print("\n")

    print("Precision = ", str(pre_store))

    print("Recall = ", str(recall_store))

    print("F1-Score = ", str(round(2.0 * (float(pre_store) * float(recall_store))/(float(pre_store) + float(recall_store)) , 3)))

    print("Accuracy = ", str(round(acc,2)))

    print("mAP = ", "{0:.2f}%".format(mAP*100))



    print("All-Ground-Truth = ", str(tp_store + fn_store))

    # cm_df.set_index('')
    # cm_df.to_csv(dir + '/matrix/confusion_matrix_'+ file +'.csv', encoding='cp949')

    """
    Plot the total number of occurences of each class in the "detection-results" folder
    """
    if draw_plot:
        window_title = "detection-results-info"
        # Plot title
        plot_title = "detection-results\n"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = output_files_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives
        draw_plot_func(
            det_counter_per_class,
            len(det_counter_per_class),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
            )

    """
    Write number of detected objects per class to output.txt
    """
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            output_file.write(text)

    """
    Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = output_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    """
    Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        x_label = "Average Precision"
        output_path = output_files_path + "/mAP.png"
        to_show = True
        plot_color = 'royalblue'
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    result.close()    

elif file == "missing" or file == "toothinfo" :

    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    GT_PATH = os.path.join(os.getcwd(), 'input', 'ground-truth', file_name)
    DR_PATH = os.path.join(os.getcwd(), 'input', 'detection-results', file_name)

    # if there are no images then no animation can be shown
    IMG_PATH = os.path.join(os.getcwd(), 'input', 'images-optional')
    if os.path.exists(IMG_PATH): 
        for dirpath, dirnames, files in os.walk(IMG_PATH):
            if not files:
                # no image files found
                args.no_animation = True
    else:
        args.no_animation = True

    # try to import OpenCV if the user didn't choose the option --no-animation
    show_animation = False
    if not args.no_animation:
        try:
            import cv2
            show_animation = True
        except ImportError:
            print("\"opencv-python\" not found, please install to visualize the results.")
            args.no_animation = True

    # try to import Matplotlib if the user didn't choose the option --no-plot
    draw_plot = False
    if not args.no_plot:
        try:
            import matplotlib.pyplot as plt
            draw_plot = True
        except ImportError:
            print("\"matplotlib\" not found, please install it to get the resulting plots.")
            args.no_plot = True


    def log_average_miss_rate(prec, rec, num_images):
        """
            log-average miss rate:
                Calculated by averaging miss rates at 9 evenly spaced FPPI points
                between 10e-2 and 10e0, in log-space.

            output:
                    lamr | log-average miss rate
                    mr | miss rate
                    fppi | false positives per image

            references:
                [1] Dollar, Piotr, et al. "Pedestrian Detection: An Evaluation of the
                State of the Art." Pattern Analysis and Machine Intelligence, IEEE
                Transactions on 34.4 (2012): 743 - 761.
        """

        # if there were no detections of that class
        if prec.size == 0:
            lamr = 0
            mr = 1
            fppi = 0
            return lamr, mr, fppi

        fppi = (1 - prec)
        mr = (1 - rec)

        fppi_tmp = np.insert(fppi, 0, -1.0)
        mr_tmp = np.insert(mr, 0, 1.0)

        # Use 9 evenly spaced reference points in log-space
        ref = np.logspace(-2.0, 0.0, num = 9)
        for i, ref_i in enumerate(ref):
            # np.where() will always find at least 1 index, since min(ref) = 0.01 and min(fppi_tmp) = -1.0
            j = np.where(fppi_tmp <= ref_i)[-1][-1]
            ref[i] = mr_tmp[j]

        # log(0) is undefined, so we use the np.maximum(1e-10, ref)
        lamr = math.exp(np.mean(np.log(np.maximum(1e-10, ref))))

        return lamr, mr, fppi

    """
    throw error and exit
    """
    def error(msg):
        print(msg)
        sys.exit(0)

    """
    check if the number is a float between 0.0 and 1.0
    """
    def is_float_between_0_and_1(value):
        try:
            val = float(value)
            if val > 0.0 and val < 1.0:
                return True
            else:
                return False
        except ValueError:
            return False


    def voc_ap(rec, prec):
        """
        --- Official matlab code VOC2012---
        mrec=[0 ; rec ; 1];
        mpre=[0 ; prec ; 0];
        for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
        end
        i=find(mrec(2:end)~=mrec(1:end-1))+1;
        ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        rec.insert(0, 0.0) # insert 0.0 at begining of list
        rec.append(1.0) # insert 1.0 at end of list
        mrec = rec[:]
        prec.insert(0, 0.0) # insert 0.0 at begining of list
        prec.append(0.0) # insert 0.0 at end of list
        mpre = prec[:]
        """
        This part makes the precision monotonically decreasing
            (goes from the end to the beginning)
            matlab: for i=numel(mpre)-1:-1:1
                        mpre(i)=max(mpre(i),mpre(i+1));
        """

        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        """
        This part creates a list of indexes where the recall changes
            matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
        """
        i_list = []
        for i in range(1, len(mrec)):
            if mrec[i] != mrec[i-1]:
                i_list.append(i) # if it was matlab would be i + 1
        """
        The Average Precision (AP) is the area under the curve
            (numerical integration)
            matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
        """
        ap = 0.0
        for i in i_list:
            ap += ((mrec[i]-mrec[i-1])*mpre[i])

        return ap, mrec, mpre


    """
    Convert the lines of a file to a list
    """
    def file_lines_to_list(path):
        # open txt file lines to a list
        with open(path) as f:
            content = f.readlines()
        # remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]
        return content

    """
    Draws text in image
    """
    def draw_text_in_image(img, text, pos, color, line_width):
        font = cv2.FONT_HERSHEY_PLAIN
        fontScale = 1
        lineType = 1
        bottomLeftCornerOfText = pos
        cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                color,
                lineType)
        text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
        return img, (line_width + text_width)

    """
    Plot - adjust axes
    """
    def adjust_axes(r, t, fig, axes):
        # get text width for re-scaling
        bb = t.get_window_extent(renderer=r)
        text_width_inches = bb.width / fig.dpi
        # get axis width in inches
        current_fig_width = fig.get_figwidth()
        new_fig_width = current_fig_width + text_width_inches
        propotion = new_fig_width / current_fig_width
        # get axis limit
        x_lim = axes.get_xlim()
        axes.set_xlim([x_lim[0], x_lim[1]*propotion])

    """
    Draw plot using Matplotlib
    # 치아인식 , 상실치아 
    """
    def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
        # sort the dictionary by decreasing value, into a list of tuples
        sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(0), reverse=True)
        # sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1)) # value DESC시에 
        # unpacking the list of tuples into two lists
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)
        # 
        if true_p_bar != "":
            """
            Special case to draw in:
                - green -> TP: True Positives (object detected and matches ground-truth)
                - red -> FP: False Positives (object detected but does not match ground-truth)
                - pink -> FN: False Negatives (object not detected but present in the ground-truth) // 미구현
            """
            fp_sorted = []
            tp_sorted = []

            for key in sorted_keys:
                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])


            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
            # add legend
            plt.legend(loc='lower right')
            """
            Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                # trick to paint multicolor with offset:
                # first paint everything and then repaint the first number
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values)-1): # largest bar
                    adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            """
            Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val) # add a space before
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                # re-set axes to show number inside the figure
                if i == (len(sorted_values)-1): # largest bar
                    adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title(window_title)
        # write classes in y axis
        tick_font_size = 12
        plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
        """
        Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height 
        top_margin = 0.15 # in percentage of the figure height
        bottom_margin = 0.05 # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title(plot_title, fontsize=14)
        plt.xlabel(x_label, fontsize='large')
        # adjust size of window
        fig.tight_layout()
        # save the plot
        fig.savefig(output_path)
        # show image
        if to_show:
            plt.show()
        # close the plot
        plt.close()

    # detection_result
    def draw_plot_func_result(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
        # sort the dictionary by decreasing value, into a list of tuples
        sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(0), reverse=True)
        # sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1)) # value DESC시에 
        # unpacking the list of tuples into two lists
        sorted_keys, sorted_values = zip(*sorted_dic_by_value)
        # 
        if true_p_bar != "":
            """
            Special case to draw in:
                - green -> TP: True Positives (object detected and matches ground-truth)
                - red -> FP: False Positives (object detected but does not match ground-truth)
                - pink -> FN: False Negatives (object not detected but present in the ground-truth) // 미구현
            """
            fp_sorted = []
            tp_sorted = []

            for key in sorted_keys:
                fp_sorted.append(dictionary[key] - true_p_bar[key])
                tp_sorted.append(true_p_bar[key])


            plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
            plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
            # add legend
            plt.legend(loc='best')
            """
            Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):

                fp_val = fp_sorted[i]
                tp_val = tp_sorted[i]
                fp_str_val = " " + str(fp_val)
                tp_str_val = fp_str_val + " " + str(tp_val)
                # trick to paint multicolor with offset:
                # first paint everything and then repaint the first number
                t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
                plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
                if i == (len(sorted_values)-1): # largest bar
                    adjust_axes(r, t, fig, axes)
        else:
            plt.barh(range(n_classes), sorted_values, color=plot_color)
            """
            Write number on side of bar
            """
            fig = plt.gcf() # gcf - get current figure
            axes = plt.gca()
            r = fig.canvas.get_renderer()
            for i, val in enumerate(sorted_values):
                str_val = " " + str(val) # add a space before
                if val < 1.0:
                    str_val = " {0:.2f}".format(val)
                t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
                # re-set axes to show number inside the figure
                if i == (len(sorted_values)-1): # largest bar
                    adjust_axes(r, t, fig, axes)
        # set window title
        fig.canvas.set_window_title(window_title)
        # write classes in y axis
        tick_font_size = 12
        plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
        """
        Re-scale height accordingly
        """
        init_height = fig.get_figheight()
        # comput the matrix height in points and inches
        dpi = fig.dpi
        height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
        height_in = height_pt / dpi
        # compute the required figure height 
        top_margin = 0.15 # in percentage of the figure height
        bottom_margin = 0.05 # in percentage of the figure height
        figure_height = height_in / (1 - top_margin - bottom_margin)
        # set new height
        if figure_height > init_height:
            fig.set_figheight(figure_height)

        # set plot title
        plt.title(plot_title, fontsize=14)
        plt.xlabel(x_label, fontsize='large')
        # adjust size of window
        fig.tight_layout()
        # save the plot
        fig.savefig(output_path)
        # show image
        if to_show:
            plt.show()
        # close the plot
        plt.close()



    """
    Create a ".temp_files/" and "output/" directory
    """
    TEMP_FILES_PATH = ".temp_files"
    if not os.path.exists(TEMP_FILES_PATH): # if it doesn't exist already
        os.makedirs(TEMP_FILES_PATH)
    output_files_path = os.path.join("output", file)

    if os.path.exists(output_files_path): # if it exist already
        # reset the output directory
        shutil.rmtree(output_files_path)
        print("output 폴더를 초기화 합니다.")

    os.makedirs(output_files_path)


    if draw_plot:
        os.makedirs(os.path.join(output_files_path , "classes"))
    if show_animation:
        os.makedirs(os.path.join(output_files_path , "images", "detections_one_by_one"))

    all_classes = ['11', '12', '13', '14', '15', '16', '17', '18', '21', '22', '23', '24', '25', '26', '27', '28', '31', '32', '33', '34', '35', '36', '37', '38', '41', '42', '43', '44', '45', '46', '47', '48', 'N']
    """
    ground-truth
        Load each of the ground-truth files into a temporary ".json" file.
        Create a list of all the class names present in the ground-truth (gt_classes).
    """
    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH)
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    for txt_file in ground_truth_files_list:
        print("GT 파일 리스트 : " , txt_file)
        
        category_id = txt_file.split(".txt", 1)[0]
        category_id = os.path.basename(os.path.normpath(category_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH)
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                        class_name, image_name,  left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                else:
                        class_name, image_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            # check if class is in the ignore list, if yes skip
            if class_name in args.ignore:
                continue
            bbox = left + " " + top + " " + right + " " +bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "image_name":image_name,  "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "image_name":image_name, "bbox":bbox, "used":False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + category_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)

    """
    Check format of the flag --set-class-iou (if used)
        e.g. check if class exists
    """
    if specific_iou_flagged:
        n_args = len(args.set_class_iou)
        error_msg = \
            '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
        if n_args % 2 != 0:
            error('Error, missing arguments. Flag usage:' + error_msg)
        # [class_1] [IoU_1] [class_2] [IoU_2]
        # specific_iou_classes = ['class_1', 'class_2']
        specific_iou_classes = args.set_class_iou[::2] # even
        # iou_list = ['IoU_1', 'IoU_2']
        iou_list = args.set_class_iou[1::2] # odd
        if len(specific_iou_classes) != len(iou_list):
            error('Error, missing arguments. Flag usage:' + error_msg)
        for tmp_class in specific_iou_classes:
            if tmp_class not in gt_classes:
                        error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
        for num in iou_list:
            if not is_float_between_0_and_1(num):
                error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

    """
    detection-results
        Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH)
    dr_files_list.sort()

    for class_index, class_name in enumerate(all_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            print("detection 파일 리스트 : " , txt_file)
            # the first time it checks if all the corresponding ground-truth files exist
            category_id = txt_file.split(".txt",1)[0]
            category_id = os.path.basename(os.path.normpath(category_id))
            temp_path = os.path.join(GT_PATH)
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, image_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    #print("match")
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "image_name": image_name ,"category_id":category_id, "bbox":bbox})
                    #print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
    Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}
    """
    TP , FP , FN , TN  저장용 변수 선언 ##########

    """
    tp_store = [] # TP값 저장 변수 
    fp_store = [] # FP값 저장 변수
    fn_store = [] # FN값 저장 변수
    tn_store = [] # TN값 임시 저장 변수
    pre_store = []
    recall_store = []
    F1_store = [] 
    class_gt = [] # 그라운드 클래스 이름 
    class_dr = [] # 추론 클래스 이름 

    dic_a = {}
    list_a = []
    # 결과를 텍스트로 저장 - /matrix/Matrix_data_ 
    dir = os.getcwd()
    if not os.path.exists(dir + '/matrix/'):
        os.makedirs(dir + '/matrix/')
    result = open(dir + '/matrix/Matrix_data_'+ file +'.txt','wt')


    gt_TP_list = [] # TP 그라운드 클래스
    dr_TP_list = [] # TP 추론 클래스
    dr_FP_list = [] # FP 추론 클래스
    gt_FP_list = [] # FP 그라운드 클래스 
    all_gt_list = [] # all 그라운드 클래스 
    grape_N_list = [] # N 그래프 리스트
    class_fp_list = [] 

    text_store = ""
    count = 11
    # open file to store the output
    with open(output_files_path + "/output.txt", 'w') as output_file:
        output_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
        for class_index, class_name in enumerate(all_classes):
            count_true_positives[class_name] = 0
            """
            Load detection-results of that class
            """
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            """
            Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            ap_list = []
            result_list = []

            start = time.time()
            print("TP/FP 구별 구간 추론시작")
            for idx, detection in enumerate(dr_data): # 추론데이터 불러오는 시점
                category_id = detection["category_id"]
                # print("category_id === ", category_id)
                # assign detection-results to ground truth object if any
                # open ground-truth with that category_id
                gt_file = TEMP_FILES_PATH + "/" + category_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [ float(x) for x in detection["bbox"].split() ] # 추론 bbox
                bb_name = detection["image_name"]
                for obj in ground_truth_data:            # 그라운드 데이터 불러오는 시점
                    # look for a class_name match
                    if (obj["class_name"] == class_name):
                        bbgt = [ float(x) for x in obj["bbox"].split() ] # ground bbox
                        bbgt_name = obj["image_name"] 
                        class_gt.append(bbgt_name)
                        if (bb_name == bbgt_name) :       
                            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:    # 기본 OVMAX 보다 높은 IOU (OV) 값이 들어올시 저장 
                                    ovmax = ov
                                    gt_match = obj # ground_truth_data 저장 
                                    print("all 그라운드 클래스 == ", gt_match['class_name'])
                                    all_gt_list.append(gt_match['class_name'])
                                    
                # assign detection as true positive/don't care/false positive
                min_overlap = MINOVERLAP  # 설정한 초기 IOU값을 가져옴 
                if specific_iou_flagged:
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        min_overlap = float(iou_list[index])

                if ovmax >= min_overlap: #  추론한 IOU값이 초기설정 IOU 보다 높을 경우. 
                    if "difficult" not in gt_match:
                            if not bool(gt_match["used"]): # ground_truth_data가 "USED"로 처리되지 않은것  
                                # true positive
                                tp[idx] = 1 # TP 배열에 1 저장 

                                # 추론된 TP 클래스 11 ~ 47 까지
                                print("추론된 TP 클래스== ", class_name)
                                dr_TP_list.append(class_name)

                                # 그라운드 TP 클래스
                                print("그라운드 TP 클래스== ", gt_match['class_name'],gt_match["image_name"])
                                gt_TP_list.append(gt_match['class_name'])

                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file, 'w') as f:
                                        f.write(json.dumps(ground_truth_data))
                                if show_animation:
                                    status = "MATCH!"
                            
                            else:
                                # false positive (multiple detection) / 0.5 보다 높은 FP
                                fp[idx] = 1 # FP 배열에 1 저장 

                                
                                if show_animation:
                                    status = "REPEATED MATCH!"
                else: 
                    # false positive / 0.5 보다 낮은 FP
                    fp[idx] = 1 # FP 배열에 1 저장 

                # fp 전체     
                fp_dict = {}    
                if fp[idx] == 1:    
                    ioumax = 0.0
                    iou_max_list = []
                    iou_max_class = []
                    _ov = 0.0

                    for obj in ground_truth_data:
                        if bb_name == obj["image_name"]:
                            if class_name != obj["class_name"]:
                                bbgt = [ float(x) for x in obj["bbox"].split() ] # ground bbox
                                
                                bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                                iw = bi[2] - bi[0] + 1
                                ih = bi[3] - bi[1] + 1
                                if iw > 0 and ih > 0:
                                    # compute overlap (IoU) = area of intersection / area of union
                                    ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                    + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                    _ov = iw * ih / ua
                                    iou_max_list.append({obj["class_name"] : _ov}) 
                                    if _ov > ioumax:    
                                        ioumax = _ov

                    class_name_gt = "N"                        
                    is_max = False

                    for a in iou_max_list:
                        for k, v in a.items():
                            if ioumax == v:
                                result_list.append(a)
                                class_name_gt = k
                                is_max = True
                        if is_max == True:
                            break

                    if class_name_gt == "N" :
                        dr_FP_list.append(class_name_gt) 
                    else:
                        dr_FP_list.append(class_name)    

                    try:
                        print("그라운드_FP 클래스 ", gt_match['class_name'],gt_match["image_name"])
                        # print("gt_match['class_name'] ==",gt_match['class_name'], class_name_gt)
                        if class_name_gt != class_name :
                            if _ov <= 0.0 :
                                gt_FP_list.append("N") 
                                grape_N_list.append("N") # N값이 저장된 그래프용 리스트 -
                            else :
                                gt_FP_list.append(class_name_gt) 
                        else :    
                            gt_FP_list.append(gt_match['class_name'])  
                    except:
                        gt_FP_list.append(class_name_gt) # gt_match['class_name'] 값이 -1로 들어왔을때.

                                
                for results in result_list:   
                    cnt_fp = 0  
                    for key, value in results.items():     
                        if fp_dict.get(key):            # 있으면
                            cnt_fp = fp_dict[key] + 1
                        else:
                            cnt_fp = 1
                        fp_dict[key] = cnt_fp    
            
            class_fp_list.append({class_name : fp_dict})
            tp_store.append({class_name : sum(tp)})
            # result.write('< '+class_name +' 치아> \n 잡힌 박스수:\t'+ str(sum(tp)+sum(fp))+'\n'+ "TP\t" + str(sum(tp)) + '\n' + "FP\t" + str(fp)+'\n')
            
            print(count , " 치아 계산 완료. ")
            print("TP/FP 구별 구간 걸린 시간 :: ",time.time()-start)
            count+=1


        gt_tmp_list = []
        dr_tmp_list = []
        all_dr_data = []
        for class_name in all_classes:
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            all_dr_data.extend([{"class_name":class_name, "image_name": drd['image_name']} for drd in dr_data])

        for gtd in ground_truth_data:  
            tmp = False
            for drd in all_dr_data:
                if drd["image_name"] == gtd["image_name"]:
                    if gtd["class_name"] == drd["class_name"]: 
                        tmp = True
                        break
            if tmp == False:
                gt_tmp_list.append(gtd["class_name"])
                dr_tmp_list.append('N')


        gt_file_list = []
        false_gt_tmp = []
        false_dr_tmp = []
        for gtd in ground_truth_data : 
            gt_file_list.append(gtd["image_name"])

        gt_file_list_set = list(set(gt_file_list)) # 중복제거

        start1 = time.time()
        for class_name in all_classes:

            if class_name != "N" :
                for image in gt_file_list_set:
                    tmp1 = False # 그라운드
                    tmp2 = False # 추론 
                    for gtd in ground_truth_data:
                        if image == gtd["image_name"]:
                            if gtd["class_name"] == class_name: # 그라운드
                                tmp1 = True
                                break     

                    for drd in all_dr_data:
                        if image == drd["image_name"]:
                            if drd["class_name"] == class_name: # 추론
                                # if image == "0001.png":
                                #     print("GT 클래스 : ", image , class_name)
                                tmp2 = True
                                break
                    
                    if tmp1 == False and tmp2 == False :
                        # if image == "0009.png":
                        #     print("GT 클래스 : ", image , class_name)
                        false_gt_tmp.append('N')
                        false_dr_tmp.append('N')                
       

        mAP = sum_AP / n_classes
        text = "mAP\t{0:.2f}%".format(mAP*100)
        text_store = text 
        output_file.write(text + "\n")
        print(text)


    N_data = []
    print("---------Ground-truth-----------------")
    print("TP 그라운드 클래스",len(gt_TP_list))
    print("FP 그라운드 클래스",len(gt_FP_list))
    print("---------Detection-result-----------")
    print("TP 추론 클래스",len(dr_TP_list))
    print("FP 추론 클래스", len(dr_FP_list))
    print("--------all ground truth -----------")
    print("all-ground-class", len(all_gt_list))
    print("-------- FN -----------")
    print("all FN ", len(all_gt_list) - len(dr_TP_list))
    gt_list_all = []
    # gt_list_all = gt_TP_list + gt_FP_list + gt_tmp_list 
    gt_list_all = gt_TP_list + gt_FP_list + gt_tmp_list + false_gt_tmp

    dr_list_all = []
    # dr_list_all = dr_TP_list + dr_FP_list + dr_tmp_list 
    dr_list_all = dr_TP_list + dr_FP_list + dr_tmp_list + false_dr_tmp
    
    print("gt_TP_list =", gt_TP_list)
    print("gt_FP_list =", gt_FP_list)
    print("dr_TP_list =", dr_TP_list)
    print("dr_FP_list =", dr_FP_list)


    print("<gt_list_all>= \n", gt_list_all)
    print("<dr_list_all>= \n", dr_list_all)

    gt_list_all_idx = list(set(gt_list_all))
    dr_list_all_col = list(set(dr_list_all))

    union_list = sorted(list(set().union(gt_list_all_idx,dr_list_all_col)))
    print(union_list)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import classification_report

    y_true = gt_list_all  # ground-truth
    y_pred = dr_list_all   # detection-result

    con = confusion_matrix(y_true, y_pred) # ground-truth , detection-result 두 리스트의 숫자가 일치해야만 실행됨. 
    print("confusion Matrix = \n", con)
    # print(type(con))

    con_df = pd.DataFrame(con, columns=union_list, index = union_list)

    add_list = list(set(all_classes).difference(set(union_list)))

    for i in add_list:
        con_df[i] = 0
        con_df.loc[i] = 0

    con_df = con_df.reindex(index=all_classes, columns=all_classes)
    N_data = con_df['N']
    print(con_df)


    con_df.to_csv(dir + '/matrix/sklearn_confusion_matrix_'+ file +'.csv', encoding='utf-8')

    from sklearn.metrics import classification_report
    print('\nClassification Report\n')
    print(classification_report(y_true, y_pred,))

    print("============ confusion matrix finish!!!!! =============")

    """
    output 파일 재생성 구간 ========================================================================================================
        
    """
    """
    ground-truth
        Load each of the ground-truth files into a temporary ".json" file.
        Create a list of all the class names present in the ground-truth (gt_classes).
    """

    # get a list with the ground-truth files
    ground_truth_files_list = glob.glob(GT_PATH)
    if len(ground_truth_files_list) == 0:
        error("Error: No ground-truth files found!")
    ground_truth_files_list.sort()
    # dictionary with counter per class
    gt_counter_per_class = {}
    counter_images_per_class = {}

    gt_files = []
    for txt_file in ground_truth_files_list:
        print("GT 파일 리스트 : " , txt_file)
        
        category_id = txt_file.split(".txt", 1)[0]
        category_id = os.path.basename(os.path.normpath(category_id))
        # check if there is a correspondent detection-results file
        temp_path = os.path.join(DR_PATH)
        if not os.path.exists(temp_path):
            error_msg = "Error. File not found: {}\n".format(temp_path)
            error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
            error(error_msg)
        lines_list = file_lines_to_list(txt_file)
        # create ground-truth dictionary
        bounding_boxes = []
        is_difficult = False
        already_seen_classes = []
        for line in lines_list:
            try:
                if "difficult" in line:
                        class_name, image_name,  left, top, right, bottom, _difficult = line.split()
                        is_difficult = True
                else:
                        class_name, image_name, left, top, right, bottom = line.split()
            except ValueError:
                error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
                error_msg += " Received: " + line
                error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
                error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
                error(error_msg)
            # check if class is in the ignore list, if yes skip
            if class_name in args.ignore:
                continue
            bbox = left + " " + top + " " + right + " " +bottom
            if is_difficult:
                bounding_boxes.append({"class_name":class_name, "image_name":image_name,  "bbox":bbox, "used":False, "difficult":True})
                is_difficult = False
            else:
                bounding_boxes.append({"class_name":class_name, "image_name":image_name, "bbox":bbox, "used":False})
                # count that object
                if class_name in gt_counter_per_class:
                    gt_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    gt_counter_per_class[class_name] = 1

                if class_name not in already_seen_classes:
                    if class_name in counter_images_per_class:
                        counter_images_per_class[class_name] += 1
                    else:
                        # if class didn't exist yet
                        counter_images_per_class[class_name] = 1
                    already_seen_classes.append(class_name)

        # dump bounding_boxes into a ".json" file
        new_temp_file = TEMP_FILES_PATH + "/" + category_id + "_ground_truth.json"
        gt_files.append(new_temp_file)
        with open(new_temp_file, 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    gt_classes = list(gt_counter_per_class.keys())
    # let's sort the classes alphabetically
    gt_classes = sorted(gt_classes)
    n_classes = len(gt_classes)


    """
    Check format of the flag --set-class-iou (if used)
        e.g. check if class exists
    """
    if specific_iou_flagged:
        n_args = len(args.set_class_iou)
        error_msg = \
            '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
        if n_args % 2 != 0:
            error('Error, missing arguments. Flag usage:' + error_msg)
        # [class_1] [IoU_1] [class_2] [IoU_2]
        # specific_iou_classes = ['class_1', 'class_2']
        specific_iou_classes = args.set_class_iou[::2] # even
        # iou_list = ['IoU_1', 'IoU_2']
        iou_list = args.set_class_iou[1::2] # odd
        if len(specific_iou_classes) != len(iou_list):
            error('Error, missing arguments. Flag usage:' + error_msg)
        for tmp_class in specific_iou_classes:
            if tmp_class not in gt_classes:
                        error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
        for num in iou_list:
            if not is_float_between_0_and_1(num):
                error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

    """
    detection-results
        Load each of the detection-results files into a temporary ".json" file.
    """
    # get a list with the detection-results files
    dr_files_list = glob.glob(DR_PATH)
    dr_files_list.sort()

    for class_index, class_name in enumerate(gt_classes):
        bounding_boxes = []
        for txt_file in dr_files_list:
            # the first time it checks if all the corresponding ground-truth files exist
            category_id = txt_file.split(".txt",1)[0]
            category_id = os.path.basename(os.path.normpath(category_id))
            temp_path = os.path.join(GT_PATH)
            if class_index == 0:
                if not os.path.exists(temp_path):
                    error_msg = "Error. File not found: {}\n".format(temp_path)
                    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-dr.py)"
                    error(error_msg)
            lines = file_lines_to_list(txt_file)
            for line in lines:
                try:
                    tmp_class_name, image_name, confidence, left, top, right, bottom = line.split()
                except ValueError:
                    error_msg = "Error: File " + txt_file + " in the wrong format.\n"
                    error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
                    error_msg += " Received: " + line
                    error(error_msg)
                if tmp_class_name == class_name:
                    #print("match")
                    bbox = left + " " + top + " " + right + " " +bottom
                    bounding_boxes.append({"confidence":confidence, "image_name": image_name ,"category_id":category_id, "bbox":bbox})
                    #print(bounding_boxes)
        # sort detection-results by decreasing confidence
        bounding_boxes.sort(key=lambda x:float(x['confidence']), reverse=True)
        with open(TEMP_FILES_PATH + "/" + class_name + "_dr.json", 'w') as outfile:
            json.dump(bounding_boxes, outfile)

    """
    Calculate the AP for each class
    """
    sum_AP = 0.0
    ap_dictionary = {}
    lamr_dictionary = {}


    # open file to store the output
    with open(output_files_path + "/output.txt", 'w') as output_file:
        output_file.write("# AP and precision/recall per class\n")
        count_true_positives = {}
    
        for class_index, class_name in enumerate(gt_classes):
            count_true_positives[class_name] = 0
            """
            Load detection-results of that class
            """
            dr_file = TEMP_FILES_PATH + "/" + class_name + "_dr.json"
            dr_data = json.load(open(dr_file))
            """
            Assign detection-results to ground-truth objects
            """
            nd = len(dr_data)
            tp = [0] * nd # creates an array of zeros of size nd
            fp = [0] * nd
            ap_list = []
            result_list = []

            for idx, detection in enumerate(dr_data): # 추론데이터 불러오는 시점
                category_id = detection["category_id"]
                # print("category_id === ", category_id)
                # assign detection-results to ground truth object if any
                # open ground-truth with that category_id
                gt_file = TEMP_FILES_PATH + "/" + category_id + "_ground_truth.json"
                ground_truth_data = json.load(open(gt_file))
                ovmax = -1
                gt_match = -1
                # load detected object bounding-box
                bb = [ float(x) for x in detection["bbox"].split() ] # 추론 bbox
                bb_name = detection["image_name"]
                for obj in ground_truth_data:            # 그라운드 데이터 불러오는 시점
                    # look for a class_name match
                    if (obj["class_name"] == class_name):
                        bbgt = [ float(x) for x in obj["bbox"].split() ] # ground bbox
                        bbgt_name = obj["image_name"] 
                        if (bb_name == bbgt_name) :       
                            bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
                            iw = bi[2] - bi[0] + 1
                            ih = bi[3] - bi[1] + 1
                            if iw > 0 and ih > 0:
                                # compute overlap (IoU) = area of intersection / area of union
                                ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
                                                + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
                                ov = iw * ih / ua
                                if ov > ovmax:    # 기본 OVMAX 보다 높은 IOU (OV) 값이 들어올시 저장 
                                    ovmax = ov
                                    gt_match = obj # ground_truth_data 저장 
                                    
                # assign detection as true positive/don't care/false positive
                min_overlap = MINOVERLAP  # 설정한 초기 IOU값을 가져옴 
                if specific_iou_flagged:
                    if class_name in specific_iou_classes:
                        index = specific_iou_classes.index(class_name)
                        min_overlap = float(iou_list[index])
                if ovmax >= min_overlap: #  추론한 IOU값이 초기설정 IOU 보다 높을 경우. 

                    if "difficult" not in gt_match:
                            if not bool(gt_match["used"]): # ground_truth_data가 "USED"로 처리되지 않은것  
                                # true positive
                                tp[idx] = 1 # TP 배열에 1 저장 
                                gt_match["used"] = True
                                count_true_positives[class_name] += 1
                                # update the ".json" file
                                with open(gt_file, 'w') as f:
                                        f.write(json.dumps(ground_truth_data))
   
                            else:
                                # false positive (multiple detection) / 0.5 보다 높은 FP
                                fp[idx] = 1 # FP 배열에 1 저장 
                else: 
                    # false positive / 0.5 보다 낮은 FP
                    fp[idx] = 1 # FP 배열에 1 저장 

            
            cumsum = 0
            for idx, val in enumerate(fp):
                fp[idx] += cumsum
                cumsum += val
            cumsum = 0
            for idx, val in enumerate(tp):
                tp[idx] += cumsum
                cumsum += val
            # print(fp)

            
            # tp , fp 를 기입해 넣어서 rec , prec에 포함시켜 voc_ap로 보냄.
            rec = tp[:]
            for idx, val in enumerate(tp):
                rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
            #Recall 출력      

            prec = tp[:]
            for idx, val in enumerate(tp):
                prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])


            ap, mrec, mprec = voc_ap(rec[:], prec[:])
            sum_AP += ap
            # our_ap = sum(prec) / len(prec)

            text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP " #class_name + " AP = {0:.2f}%".format(ap*100)
            """
            Write to output.txt
            """
            rounded_prec = [ '%.3f' % elem for elem in prec ]
            rounded_rec = [ '%.3f' % elem for elem in rec ]
            output_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall :" + str(rounded_rec) + "\n\n")
            

            if not args.quiet:
                print(text)
            ap_dictionary[class_name] = ap

            n_images = counter_images_per_class[class_name]
            lamr, mr, fppi = log_average_miss_rate(np.array(prec), np.array(rec), n_images)
            lamr_dictionary[class_name] = lamr


            """
            Draw plot
            """
            if draw_plot:
                plt.plot(rec, prec, '-o')
                # add a new penultimate point to the list (mrec[-2], 0.0)
                # since the last line segment (and respective area) do not affect the AP value
                area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
                area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
                plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
                # set window title
                fig = plt.gcf() # gcf - get current figure
                fig.canvas.set_window_title('AP ' + class_name)
                # set plot title
                plt.title('class: ' + text)
                #plt.suptitle('This is a somewhat long figure title', fontsize=16)
                # set axis titles
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                # optional - set axes
                axes = plt.gca() # gca - get current axes
                axes.set_xlim([0.0,1.0])
                axes.set_ylim([0.0,1.05]) # .05 to give some extra space

                # save the plot
                fig.savefig(output_files_path + "/classes/" + class_name + ".png")
                plt.cla() # clear axes for next plot
        """
        Count total of detection-results
        """
        # iterate through all the files
        det_counter_per_class = {}
        for txt_file in dr_files_list:
            # get lines to list
            lines_list = file_lines_to_list(txt_file)
            for line in lines_list:
                class_name = line.split()[0]
                # check if class is in the ignore list, if yes skip
                if class_name in args.ignore:
                    continue
                # count that object
                if class_name in det_counter_per_class:
                    det_counter_per_class[class_name] += 1
                else:
                    # if class didn't exist yet
                    det_counter_per_class[class_name] = 1

        dr_classes = list(det_counter_per_class.keys())

        # N ap 계산. 단일 ap 이므로 precision 
        N_sorted = N_data.values.tolist()
        # N_sorted = N_data
        N_fp_sum = 0
        for i in N_sorted:
            if i != N_sorted[-1]:
                N_fp_sum += i
        N_tp_sum = int(N_sorted[-1])

        N_ap = N_tp_sum / (N_tp_sum + N_fp_sum)     

        # output_file.write("\n# mAP of all classes\n")
        mAP = (sum_AP+N_ap) / (n_classes + 1)
        text = "mAP\t{0:.2f}%".format(mAP*100)
        text_store = text 
        output_file.write(text + "\n")
        print(text)

        
    """
    Plot the total number of occurences of each class in the ground-truth
    """
    if draw_plot:
        window_title = "ground-truth-info"
        plot_title = "ground-truth\n"
        x_label = "Number of objects per class"
        output_path = output_files_path + "/ground-truth-info.png"
        to_show = False
        plot_color = 'forestgreen'
        draw_plot_func(
            gt_counter_per_class,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            '',
            )

    """
    Finish counting true positives
    """
    for class_name in dr_classes:
        # if class exists in detection-result but not in ground-truth then there are no true positives in that class
        if class_name not in gt_classes:
            count_true_positives[class_name] = 0


    """ ######## TP + FP +( FN + TN ) = Ground truth!!! ####### 
    Plot the total number of occurences of each class in the "detection-results" folder
    """
    if draw_plot:
        window_title = "detection-results-info"
        # Plot title
        plot_title = "detection-results\n"
        # end Plot title
        x_label = "Number of objects per class"
        output_path = output_files_path + "/detection-results-info.png"
        to_show = False
        plot_color = 'forestgreen'
        true_p_bar = count_true_positives

        N_sorted = N_data.values.tolist()
        # N_sorted = N_data
        N_fp_sum = 0
        for i in N_sorted:
            if i != N_sorted[-1]:
                N_fp_sum += i

        # print(N_fp_sum)
        N_count = int(N_sorted[-1]) + N_fp_sum       

        det_counter_per_class['N'] = N_count
        true_p_bar['N'] = int(N_sorted[-1])
        # print(det_counter_per_class)
        # print(true_p_bar)

        draw_plot_func_result(
            det_counter_per_class,
            len(det_counter_per_class),
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            true_p_bar
            )

    n_classes = n_classes + 1
    """
    Draw log-average miss rate plot (Show lamr of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "lamr"
        plot_title = "log-average miss rate"
        x_label = "log-average miss rate"
        output_path = output_files_path + "/lamr.png"
        to_show = False
        plot_color = 'royalblue'

        lamr_dictionary['N'] = 1 - N_ap
        draw_plot_func(
            lamr_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )

    """
    Draw mAP plot (Show AP's of all classes in decreasing order)
    """
    if draw_plot:
        window_title = "mAP"
        plot_title = "mAP = {0:.2f}%".format(mAP*100)
        x_label = "Average Precision"
        output_path = output_files_path + "/mAP.png"
        to_show = False
        plot_color = 'royalblue'

        ap_dictionary['N'] = N_ap
        print("ap_dictionary== ", ap_dictionary)
        draw_plot_func(
            ap_dictionary,
            n_classes,
            window_title,
            plot_title,
            x_label,
            output_path,
            to_show,
            plot_color,
            ""
            )
    """
    Write number of detected objects per class to output.txt
    """
    with open(output_files_path + "/output.txt", 'a') as output_file:
        output_file.write("\n# Number of detected objects per class\n")
        for class_name in sorted(dr_classes):
            n_det = det_counter_per_class[class_name]
            text = class_name + ": " + str(n_det)
            text += " (tp:" + str(count_true_positives[class_name]) + ""
            text += ", fp:" + str(n_det - count_true_positives[class_name]) + ")\n"
            output_file.write(text)        

    
    # ========== score 계산 =========================================================================

    if file == "toothinfo":
        # 치아식별
        data = pd.read_csv(r"matrix/sklearn_confusion_matrix_toothinfo.csv")
    elif file == "missing":    
        # 상실치아
        data = pd.read_csv(r"matrix/sklearn_confusion_matrix_missing.csv")
    
    # 0열의 class 배열로 저장 
    classes = data.to_numpy()[:, 0].tolist()
    # print(len(classes)) # 32
    
    # 0열의 class 지우기
    confuse = data.to_numpy()[:, 1:]
    
    # 0. Confusion Matrix는 1 X 1024 형태
    confusion = np.reshape(confuse, (-1))
    
    TP = []
    FP = [[0]*len(classes) for i in range(len(classes))]
    FN = [[0]*len(classes) for i in range(len(classes))]
     
     
    # 1. TP는 1 X 32 형태
    for index, value in enumerate(confusion):
        if int(index) % (len(classes)+1) == 0:
            TP.append(value)
             
    print(f"TP는 {TP}")
    
    
    # 2. FP는 32 X 32 형태 (TP 부분은 0으로 채워짐)
    i = 0
    count = 0
    
    for i in range(len(classes)):         # 카테고리 인덱스
        for index, value in enumerate(confusion):   # confusion matrix의 인덱스
            if count <= len(classes):                         # FN[카테고리 인덱스][count]
       
                if count != i:
                    if index == len(classes)*count + i:
                        FP[i][count] = value
                else:
                    FP[i][count] = 0
                     
                    if index % (len(classes)+1) == 0:
                        pass                        # TP값
                 
                if (index - i) % len(classes) == 0:
                    count += 1
                     
                if index == ((len(classes) ** 2) - 1):
                    count = 0
     
    print(f"FP는 {FP}")
    
    
    # 3. FN은 32 X 32 형태 (TP 부분은 0으로 채워짐)
    i = 0
    num = 0
     
    if i < len(classes):
        if num < len(classes):
            for index, value in enumerate(confusion):
                cate_id = i
      
                if index == cate_id*len(classes) + num:
                    if num == cate_id:
                        FN[i][num] = 0
                    else:
                        FN[i][num] = value
                         
                    num += 1
                     
                if num == len(classes):
                    num = 0
                    i += 1
                     
                if i == len(classes):
                    break
     
    print(f"FN는 {FN}")
    
    
    # 4. Accuracy = 모든 TP의 합 / Confusion Matrix의 모든 cell의 합
    Accuracy = sum(TP) / sum(confusion)
    print(f"Accuracy = {Accuracy}\n")
    
    
    # 5. 각 클래스별 FP합    => 길이 32인 리스트
    sum_fp = []
    for i,fp in enumerate(FP):
       sum_fp.append(sum(fp))
    # print(f"{len(sum_fp)}개의  sum_fp {sum_fp}") # 32, [32개 value]
    
       
    # 6. 각 클래스별 FN합    => 길이 32인 리스트
    sum_fn = []
    for i,fn in enumerate(FN):
        sum_fn.append(sum(fn))
    # print(f"{len(sum_fn)}개의  sum_fn은 {sum_fn}") # 32, [32개 value]
    
    
    # new_classes = classes.copy()
    ############ MISSING - 정답, 추론에 모두 없는 치아 번호 제외 #########
    none_class = []
    for i in range(len(classes)): 
        if TP[i] == 0 and sum_fp[i] == 0 and sum_fn[i] == 0:
            # none배열에 추가
            none_class.append(i)
    
    for index in list(reversed(none_class)):
        # TP, FP, FN에서 해당 카테고리 데이터 삭제
        del TP[index]
        del sum_fp[index]
        del sum_fn[index]
        # none class 그래프 축에서 제외
        del classes[index]
    
    
    classes = list(reversed(classes))  

    
    # 7. Precision = 해당 클래스의 TP / (TP + 해당 클래스의 모든 FP의 합) = 해당 클래스의 TP / 해당 클래스 열의 모든 cell의 합
    # 각 클래스별 precision 구하기
    precision = []
    for i,tp in enumerate(TP):
        for j,fp in enumerate(sum_fp):
            if i == j:
    #             pre = tp/(tp+fp) #### 0 error
                if tp == 0 and fp == 0: # 분모가 0일 때
                    pre = 0
                else:
                    pre = tp/(tp+fp)
        precision.append(pre)
        
    print(f"{len(precision)}개의  precision은 {precision}\n") # 32, [32개 value]
    
    
    # 8. Recall = 해당 클래스의 TP / (TP + 해당 클래스의 모든 FN의 합)
    #            = 해당 클래스의 TP / 해당 클래스 행의 모든 cell의 합
    # 각 클래스별 recall 구하기
    recall = []
    for i,tp in enumerate(TP):
        for j,fn in enumerate(sum_fn):
            if i == j:
    #             rec = tp/(tp+fn) #### 0 error
                if tp == 0 and fn == 0: # 분모가 0일 때
                    rec = 0
                else:
                    rec = tp/(tp+fn)
        recall.append(rec)
    print(f"{len(recall)}개의 recall은 {recall}")
    
        
    # 9. F1 score = 2 X [(Precision X Recall) / (Precision + Recall)]
    # 각 클래스별 F1-score 구하기
    F1_scores = []
    for i, pre in enumerate(precision):
        for j, rec in enumerate(recall):
            if i == j:
                if pre == 0 and rec == 0: # 분모가 0일 때
                    f1 = 0
                else:
                    f1 = 2*((pre * rec) / (pre + rec))
        F1_scores.append(f1)
    
    print(f"{len(F1_scores)}개의 f1-score는  {F1_scores}\n")
    
        
    # 10-1. average Precision = 모든 클래스의 Precision의 합 / 카테고리 갯수 = 모든 클래스의 precision의 합 / 32
    avg_precision = sum(precision) / (len(classes) - len(none_class))
    print(f"Average precision은  {avg_precision}\n")
    
    
    # 10-2. average Recall = 모든 클래스의 Recall의 합 / 카테고리 갯수 = 모든 클래스의 Recall의 합 / 32
    avg_recall = sum(recall) / (len(classes) - len(none_class))
    print(f"Average recall = {avg_recall}\n")
    
    
    # 10-3. Total F1 score = 2 X [(Precision X Recall) / (Precision + Recall)]
    total_f1 = 2*((avg_precision * avg_recall) / (avg_precision + avg_recall))
    print(f"total_f1 = {total_f1}")
    
    
    # csv로 결과 저장
    if file == "toothinfo":
        # 치아식별
        data_csv = pd.read_csv(r"matrix/sklearn_confusion_matrix_toothinfo.csv")
    elif file == "missing":    
        # 상실치아
        data_csv = pd.read_csv(r"matrix/sklearn_confusion_matrix_missing.csv")    

    
    # TP, FP, FN이 0인 인덱스의 precision, recall, f1score = 0
    if len(precision) < (len(classes)+1):
    
        precision_add = precision.copy()
        recall_add = recall.copy()
        F1_scores_add = F1_scores.copy()
         
        for i,val in enumerate(none_class):
            precision_add.insert(val, 0)
            recall_add.insert(val, 0)
            F1_scores_add.insert(val, 0)
         
    
        data_csv['Precision'] = precision_add
        data_csv['Recall'] = recall_add
        data_csv['F1-score'] = F1_scores_add
        
    # 33 x 33일 때
    else:
        data_csv['Precision'] = precision
        data_csv['Recall'] = recall
        data_csv['F1-score'] = F1_scores
        
    col_name = ['Total F1-score', 'Accuracy']

    total = [total_f1]
    acc = [Accuracy]
    new_arr = []
    new_arr.append(total)
    new_arr.append(acc)
    new_t = pd.DataFrame(new_arr).transpose()
    new_t.columns = col_name

    
    final_df = pd.merge(data_csv, new_t, how="outer", left_index=True, right_index=True)
    
    # data_csv.to_csv('Output.csv', index=False)
    final_df.to_csv('matrix/'+file+'_Output.csv', index=False)
    
    # 그래프
    f1_reverse = list(reversed(F1_scores))
    
    plt.barh(range(len(F1_scores)), f1_reverse)
    plt.title(f"F1-Score of all classes : {round(total_f1, 4)}")
    plt.xlabel('F1-Score')
    plt.ylabel('classes')
    plt.yticks(np.arange(len(classes)), classes)
    for i, val in enumerate(f1_reverse):
        plt.text(val, i, str(round(val,4)), color='black', va='center')
    
    plt.savefig("output/"+file+"/"+file+"_f1_score.png")
    plt.show()

    plt.close()
    print("Finished")
    

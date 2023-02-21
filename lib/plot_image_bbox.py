import argparse
import os
import json
import numpy as np
from utils import add_bbox_to_image
import cv2
from BoundingBox import BoundingBox

if __name__ == '__main__':


    parser = argparse.ArgumentParser(description='Description of your script')

    parser.add_argument('--jsonf', type=str, default='bbox_results_for_each_image.json', help='Path to bbox results image level file')
    parser.add_argument('--image_list', type=str, default=None, help='Path to the list of images, use "," as seperator')
    parser.add_argument('--image2path', type=str, default='image_id_to_abs_path.json', help='Path to bbox results image level file')

    parser.add_argument('--save_path', type=str, default='', help='Path to save output files')
    parser.add_argument('--random', type=int, default=10, help='Use random mode if random > 0')
    parser.add_argument('--random_seed', type=int, default=7)
    args = parser.parse_args()

    jsonf = next(filter(lambda x: os.path.isfile(x), [args.jsonf, os.path.join(args.save_path, args.jsonf)]))
    assert jsonf is not None


    with open(next(filter(lambda x: os.path.isfile(x), [args.image2path, os.path.join(args.save_path, args.image2path)])), 'rt') as f:
        image2abspath = json.load(f)
        print('image to path dict', len(image2abspath), next(iter(image2abspath.items())))


    with open(jsonf, 'rt') as f:
        bbox_results_dict = json.load(f)
        print('bboxes results', len(bbox_results_dict), next(iter(bbox_results_dict.items())))

    images = None
    if args.image_list is not None:
        images = list(args.image_list.split(','))
    
    if args.random > 0:
        assert args.image_list is None
        np.random.seed(args.random_seed)
        images = np.random.choice(sorted(list(bbox_results_dict.keys())), size=args.random, replace=False)
        images = list(images)

    print('plotting images', len(images))
    
    for image_name in images:
        
        bboxes = bbox_results_dict[image_name]
        abs_path = image2abspath[image_name]
        
        image = cv2.imread(abs_path, cv2.IMREAD_COLOR)

        for bb in bboxes:
            coord = bb['bbox']

            if bb['type'] == 'gt':
                image = add_bbox_to_image(image, coord[0], coord[1], coord[2], coord[3], color=(0, 255, 0))  # green
            elif bb['tp'] > 0:
                image = add_bbox_to_image(image, coord[0], coord[1], coord[2], coord[3], color=(0, 0, 255), label='TP')
            else:  # if detection
                image = add_bbox_to_image(image, coord[0], coord[1], coord[2], coord[3], color=(255, 0, 0), label='FP')  # red

        cv2.imwrite(os.path.join(args.save_path, image_name + '.jpg'), image)
    


    
    



import json
import os
import cv2

COCO_INFO = {"year":2020,"version":"1","description":"Exported using VGG Image Annotator (http://www.robots.ox.ac.uk/~vgg/software/via/)","contributor":"","url":"http://www.robots.ox.ac.uk/~vgg/software/via/","date_created":"Fri Apr 17 2020 10:41:28 GMT+0800 (China Standard Time)"}
COCO_LICENSES = [{"id":1,"name":"Unknown","url":""}]

def transfer_via_to_coco(via_anno_file, image_dir, coco_anno_file):

    
    coco_annotation_dict = {}
    coco_annotation_dict["info"] = COCO_INFO
    coco_annotation_dict["licenses"] = COCO_LICENSES
    coco_annotation_dict["images"] = []
    coco_annotation_dict["annotations"] = []
    coco_annotation_dict["categories"] = [{"id":1, "name":"panda", "supercategory":"animal"}]
    

    with open(via_anno_file, 'r') as f:
        imgs_anns = json.load(f)
    
    annotation_idx = 0
    for image_idx, anno_dict in enumerate(imgs_anns.values()):
        #print(anno_dict)
        filename = anno_dict["filename"]
        regions = anno_dict["regions"]
        
        # parse images record, 
        # - keys: 
        #             id
        #          width
        #         height
        #      file_name
        #    license(id) 
        #  data_captured
        image_file = os.path.join(image_dir, filename)
        image = cv2.imread(image_file)
        height, width, _ = image.shape
        image_record = {"id": image_idx,
                       "width": width,
                       "height": height,
                       "file_name":filename,
                       "license":1,
                       "date_captured":""}
        coco_annotation_dict["images"].append(image_record)
        
        # parse annotations record from regions, 
        # - keys: 
        #       id, 
        #       image_id, 
        #       category_id, 
        #       segementation:[[x1,x2,..],[y1,y2,..]], 
        #       bbox:[(left_top_x, left_top_y, width, height)], 
        #       iscrowd: 0,1
        for i in range(len(regions)):
            region = regions[i]
            shape_attributes_dict = region["shape_attributes"]   # region's ploy points
            region_attributes_dict = region["region_attributes"] # region's label, such as panda
            #for shape_attribute, region_attribute in zip(shape_attributes_dict, region_attributes_dict):
            #    print('shape_attribute:', shape_attribute)
            all_points_x = shape_attributes_dict["all_points_x"]
            all_points_y = shape_attributes_dict["all_points_y"]

            #segmentation = [all_points_x, all_points_y]
            poly = [(x, y) for x, y in zip(all_points_x, all_points_y)]
            poly = [p for x in poly for p in x]
            
            
            bbox_lefttop_x = min(all_points_x)
            bbox_lefttop_y = min(all_points_y)
            bbox_width = max(all_points_x) - min(all_points_x)
            bbox_height = max(all_points_y) - min(all_points_y)
            bbox = [bbox_lefttop_x, bbox_lefttop_y, bbox_width, bbox_height]
            category_id = int(region_attributes_dict["object"]) # key 'object' is decided by you in VIA, can be any thing
            annotation_record = {"id":annotation_idx,
                                "image_id": image_idx,
                                "category_id":category_id,
                                "segmentation":[poly],
                                "bbox":bbox,
                                "area": bbox_height * bbox_width,
                                "iscrowd":0}
            coco_annotation_dict["annotations"].append(annotation_record)
            annotation_idx += 1

    with open(coco_anno_file, 'w') as f:
        json.dump(coco_annotation_dict, f)

if __name__ == "__main__":
    
    for phase in ["train", "val"]:
        anno = "panda_coco/annotations/via_panda_{}.json".format(phase)
        image_dir = "panda_coco/{}".format(phase)
        output = "panda_coco/annotations/panda_{}.json".format(phase)
        transfer_via_to_coco(anno, image_dir, output)
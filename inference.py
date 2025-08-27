import argparse
from predictor import VisualizationDemo
import torch
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
import os
def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # To use demo for Panoptic-DeepLab, please uncomment the following two lines.
    # from detectron2.projects.panoptic_deeplab import add_panoptic_deeplab_config  # noqa
    # add_panoptic_deeplab_config(cfg)
    cfg.merge_from_file(args.config)#yaml file
    cfg.MODEL.WEIGHTS = args.weights#pth file
    # opts=[]
    confidence_threshold=0.5
    cfg.MODEL.DEVICE="cuda" if torch.cuda.is_available() else "cpu"
    # cfg.merge_from_list(opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
    return cfg

def main(args,image_path):
    cfg = setup_cfg(args)
    demo = VisualizationDemo(cfg)
    img = read_image(image_path, format="BGR")
    predictions, visualized_output = demo.run_on_image(img)
    
    print(predictions)
    print("detected {} instances".format(len(predictions["instances"])))
    
    if not os.path.exists("outputs"):
        os.makedirs("outputs")
    visualized_output.save(f"outputs/{image_path.split('/')[-1].split('.')[0]}.jpg")
    # return predictions, visualized_output

if __name__ == "__main__":
    # yaml_path="/home/juejue34589/Summer-Training-Week3/resnet_24000_city/config.yaml"
    # model_path="/home/juejue34589/Summer-Training-Week3/resnet_24000_city/model_final.pth"
    # image_path="/home/juejue34589/Summer-Training-Week3/dataset/Cityscapes_dataset/VOC2007/JPEGImages/zurich_000118_000019_leftImg8bit.png"
    yaml_path="resnet_3000_foggy/config.yaml"
    model_path="resnet_3000_foggy/model_final.pth"
    image_path="dataset/foggy_cityscape/VOC2007/JPEGImages/source_aachen_000049_000019_leftImg8bit.jpg"
    parser=argparse.ArgumentParser(description="Detectron2 demo for builtin models")
    parser.add_argument("--config",type=str,default=yaml_path)
    parser.add_argument("--weights",type=str,default=model_path)
    parser.add_argument("--image_path",type=str,default=image_path)
    args=parser.parse_args()

    main(args,image_path)
# https://github.com/ultralytics/ultralytics/issues/1429#issuecomment-1519239409

from pathlib import Path
import torch
import argparse
import numpy as np
import cv2
import os
from types import SimpleNamespace

from event_manager.yolo_tracking.boxmot.tracker_zoo import create_tracker
from ultralytics.yolo.engine.model import YOLO, TASK_MAP

from ultralytics.yolo.utils import LOGGER, SETTINGS, colorstr, ops, is_git_dir, IterableSimpleNamespace
from ultralytics.yolo.utils.checks import check_imgsz, print_args
from ultralytics.yolo.utils.files import increment_path
from ultralytics.yolo.engine.results import Boxes
from ultralytics.yolo.data.utils import VID_FORMATS

from event_manager.yolo_tracking.examples.multi_yolo_backend import MultiYolo
from event_manager.yolo_tracking.examples.utils import write_MOT_results

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0]  # repo root absolute path
EXAMPLES = FILE.parents[0]  # examples absolute path
WEIGHTS = EXAMPLES / 'weights'


class CustomTrack():

    def __init__(self):

        self.default_dictionary = {
        "yolo_model" : WEIGHTS / 'yolov8n.pt',
        "reid_model" : WEIGHTS / 'mobilenetv2_x1_4_dukemtmcreid.pt',
        "tracking_method" : 'deepocsort',
        "source" : '0',
        "imgsz" : [640],
        "conf" : 0.5,
        "iou" : 0.7,
        "device" : "",
        "show" : False,
        "save" : False,
        "classes" : [2,3],
        "project" : EXAMPLES / "runs" / "track",
        "name" : "exp",
        "exist_ok" : True,
        "half" : False,
        "vid_stride" : 1,
        "hide_label" : True,
        "hide_conf" : True,
        "save_txt" : False,
    }
        #print(f"Default options: {self.default_dictionary}")

        self.tracker_predictions = {}


        return

    def on_predict_start(self, predictor):
        predictor.trackers = []
        predictor.tracker_outputs = [None] * predictor.dataset.bs
        predictor.args.tracking_config = \
            ROOT /\
            'boxmot' /\
            self.default_dictionary["tracking_method"] /\
            'configs' /\
            (self.default_dictionary["tracking_method"] + '.yaml')
        for i in range(predictor.dataset.bs):
            tracker = create_tracker(
                predictor.args.tracking_method,
                predictor.args.tracking_config,
                predictor.args.reid_model,
                predictor.device,
                predictor.args.half
            )
            predictor.trackers.append(tracker)


    @torch.no_grad()
    def run(self, args):
        
        model = YOLO(args['yolo_model'] if 'v8' in str(args['yolo_model']) else 'yolov8n')
        overrides = model.overrides.copy()
        model.predictor = TASK_MAP[model.task][3](overrides=overrides, _callbacks=model.callbacks)
        
        # extract task predictor
        predictor = model.predictor

        # combine default predictor args with custom, preferring custom
        combined_args = {**predictor.args.__dict__, **args}
        # overwrite default args
        predictor.args = IterableSimpleNamespace(**combined_args)

        # setup source and model
        if not predictor.model:
            predictor.setup_model(model=model.model, verbose=False)
        predictor.setup_source(predictor.args.source)
        
        predictor.args.imgsz = check_imgsz(predictor.args.imgsz, stride=model.model.stride, min_dim=2)  # check image size
        predictor.save_dir = increment_path(Path(predictor.args.project) / predictor.args.name, exist_ok=predictor.args.exist_ok)
        
        # Check if save_dir/ label file exists
        if predictor.args.save or predictor.args.save_txt:
            (predictor.save_dir / 'labels' if predictor.args.save_txt else predictor.save_dir).mkdir(parents=True, exist_ok=True)
        # Warmup model
        if not predictor.done_warmup:
            predictor.model.warmup(imgsz=(1 if predictor.model.pt or predictor.model.triton else predictor.dataset.bs, 3, *predictor.imgsz))
            predictor.done_warmup = True
        predictor.seen, predictor.windows, predictor.batch, predictor.profilers = 0, [], None, (ops.Profile(), ops.Profile(), ops.Profile(), ops.Profile())
        predictor.add_callback('on_predict_start', self.on_predict_start)
        predictor.run_callbacks('on_predict_start')
        
        model = MultiYolo(
            model=model.predictor.model if 'v8' in str(args['yolo_model']) else args['yolo_model'],
            device=predictor.device,
            args=predictor.args
        )
        for frame_idx, batch in enumerate(predictor.dataset):
            predictor.run_callbacks('on_predict_batch_start')
            predictor.batch = batch
            path, im0s, vid_cap, s = batch

            n = len(im0s)
            predictor.results = [None] * n
            
            # Preprocess
            with predictor.profilers[0]:
                im = predictor.preprocess(im0s)

            # Inference
            with predictor.profilers[1]:
                preds = model(im, im0s)

            # Postprocess moved to MultiYolo
            with predictor.profilers[2]:
                predictor.results = model.postprocess(path, preds, im, im0s, predictor)
            predictor.run_callbacks('on_predict_postprocess_end')
            
            # Visualize, save, write results
            n = len(im0s)
            for i in range(n):
                
                if predictor.dataset.source_type.tensor:  # skip write, show and plot operations if input is raw tensor
                    continue
                p, im0 = path[i], im0s[i].copy()
                p = Path(p)
                
                with predictor.profilers[3]:
                    # get raw bboxes tensor
                    dets = predictor.results[i].boxes.data

                    # get tracker predictions
                    predictor.tracker_outputs[i] = predictor.trackers[i].update(dets.cpu().detach(), im0)
                predictor.results[i].speed = {
                    'preprocess': predictor.profilers[0].dt * 1E3 / n,
                    'inference': predictor.profilers[1].dt * 1E3 / n,
                    'postprocess': predictor.profilers[2].dt * 1E3 / n,
                    'tracking': predictor.profilers[3].dt * 1E3 / n
                }

                # filter boxes masks and pose results by tracking results
                model.filter_results(i, predictor)
                # overwrite bbox results with tracker predictions
                model.overwrite_results(i, im0.shape[:2], predictor)
                
                self.tracker_predictions[os.path.basename(p)] = predictor.tracker_outputs[i]
                
                # write inference results to a file or directory   
                if predictor.args.verbose or predictor.args.save or predictor.args.save_txt or predictor.args.show:
                    s += predictor.write_results(i, predictor.results, (p, im, im0))
                    predictor.txt_path = Path(predictor.txt_path)
                    
                    # write MOT specific results
                    if predictor.args.source.endswith(VID_FORMATS):
                        predictor.MOT_txt_path = predictor.txt_path.parent / p.stem
                    else:
                        # append folder name containing current img
                        predictor.MOT_txt_path = predictor.txt_path.parent / p.parent.name
                        
                    if predictor.tracker_outputs[i].size != 0 and predictor.args.save_txt:
                        write_MOT_results(
                            predictor.MOT_txt_path,
                            predictor.results[i],
                            frame_idx,
                            i,
                        )

                # display an image in a window using OpenCV imshow()
                if predictor.args.show and predictor.plotted_img is not None:
                    predictor.show(p.parent)

                # save video predictions
                if predictor.args.save and predictor.plotted_img is not None:
                    predictor.save_preds(vid_cap, i, str(predictor.save_dir / p.name))

            predictor.run_callbacks('on_predict_batch_end')

            # print time (inference-only)
            if predictor.args.verbose:
                LOGGER.info(f'{s}YOLO {predictor.profilers[1].dt * 1E3:.1f}ms, TRACKING {predictor.profilers[3].dt * 1E3:.1f}ms')

        # Release assets
        if isinstance(predictor.vid_writer[-1], cv2.VideoWriter):
            predictor.vid_writer[-1].release()  # release final video writer

        # Print results
        if predictor.args.verbose and predictor.seen:
            t = tuple(x.t / predictor.seen * 1E3 for x in predictor.profilers)  # speeds per image
            LOGGER.info(f'Speed: %.1fms preprocess, %.1fms inference, %.1fms postprocess, %.1fms tracking per image at shape '
                        f'{(1, 3, *predictor.args.imgsz)}' % t)
        if predictor.args.save or predictor.args.save_txt or predictor.args.save_crop:
            nl = len(list(predictor.save_dir.glob('labels/*.txt')))  # number of labels
            s = f"\n{nl} label{'s' * (nl > 1)} saved to {predictor.save_dir / 'labels'}" if predictor.args.save_txt else ''
            LOGGER.info(f"Results saved to {colorstr('bold', predictor.save_dir)}{s}")

        predictor.run_callbacks('on_predict_end')

        return 
        

    


# def main(opt):
#     run(vars(opt))


# if __name__ == "__main__":
#     opt = parse_opt()
#     main(opt)
import os
import time
from threading import Thread
from queue import Queue

import cv2
import numpy as np
import torch
import torch.multiprocessing as mp

from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.pPose_nms import pose_nms, write_json

import requests
import json

import random

DEFAULT_VIDEO_SAVE_OPT = {
    'savepath': 'examples/res/1.mp4',
    'fourcc': cv2.VideoWriter_fourcc(*'mp4v'),
    'fps': 25,
    'frameSize': (640, 480)
}

EVAL_JOINTS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

JiuShiTokenExpireTime = 10000
JiuShiStopTime = 5
JiuShiCommonSoundTime = 2

Sounds = ["买一个啦", "天气太热，买瓶可乐", "别看了，买一个吧", "好东西就得买", "我有好吃的，也有好喝的"]
CommandSounds = ["我是一台售货车哦", "我有零食", "我有可乐"]

class DataWriter():
    def __init__(self, cfg, opt, save_video=False,
                 video_save_opt=DEFAULT_VIDEO_SAVE_OPT,
                 queueSize=1024, inCameraIndex = -1):
        self.cfg = cfg
        self.opt = opt
        self.video_save_opt = video_save_opt

        self.eval_joints = EVAL_JOINTS
        self.save_video = save_video
        self.heatmap_to_coord = get_func_heatmap_to_coord(cfg)
        self.camera_idx = inCameraIndex
        self.curJiuShiTokenTime = JiuShiTokenExpireTime
        self.jiushiToken = ""
        self.lastRecordTime = 0
        self.isStopping = False
        self.accStopTime = 0.0
        self.accCommonSoundTime = 0.0
        self.stopSignal = False
        # initialize the queue used to store frames read from
        # the video file
        if opt.sp:
            self.result_queue = Queue(maxsize=queueSize)
        else:
            self.result_queue = mp.Queue(maxsize=queueSize)

        if opt.save_img:
            if not os.path.exists(opt.outputpath + '/vis'):
                os.mkdir(opt.outputpath + '/vis')

        if opt.pose_flow:
            from trackers.PoseFlow.poseflow_infer import PoseFlowWrapper
            self.pose_flow_wrapper = PoseFlowWrapper(save_path=os.path.join(opt.outputpath, 'poseflow'))

        if self.opt.save_img or self.save_video or self.opt.vis or self.opt.vis_fast:
            loss_type = self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss')
            num_joints = self.cfg.DATA_PRESET.NUM_JOINTS
            if loss_type == 'MSELoss':
                self.vis_thres = [0.4] * num_joints
            elif 'JointRegression' in loss_type:
                self.vis_thres = [0.05] * num_joints
            elif loss_type == 'Combined':
                if num_joints == 68:
                    hand_face_num = 42
                else:
                    hand_face_num = 110
                self.vis_thres = [0.4] * (num_joints - hand_face_num) + [0.05] * hand_face_num

        self.use_heatmap_loss = (self.cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss')

    def start_worker(self, target):
        if self.opt.sp:
            p = Thread(target=target, args=())
        else:
            p = mp.Process(target=target, args=())
        # p.daemon = True
        p.start()
        return p

    def start(self):
        # start a thread to read pose estimation results per frame
        self.result_worker = self.start_worker(self.update)
        return self

    def GetJiuShiToken(self):
        if self.curJiuShiTokenTime >= JiuShiTokenExpireTime:
            url = "https://auth.zelostech.com.cn/app/accessToken"
            body = {
                "appId": "oc6999f41e0b6464baebe88c26124c59d",
                "appKey": "ZmFjMDRmZjgtYWE5Yi00MDQ0LWJmODYtNGMxNTc4ZmVlOTUz"
            }

            headers = {"Content-Type": "application/json"}
            response = requests.post(url, data=json.dumps(body), headers=headers)
            parseJsonData = response.json()
            try:
                token = parseJsonData['data']['token']
                self.jiushiToken = token
                print("Token:", token)
            except KeyError as e:
                print(f"token is lost: {e}")
            self.curJiuShiTokenTime = 0
        else:
            self.curJiuShiTokenTime += 1

    def TellJiuShiStop(self):
        url = "https://gateway.zelostech.com.cn/business-server/open-apis/vehicle/command"
        body = {
            "vehicleName": "ZL01351",
            "commandType": "EMERGENCY_STOP",
            "userId": "15",
            "userName": "testUser",
            "source": "34b5c696d54941e217938c590d45b598"
        }

        headers = {"Content-Type": "application/json", 
            "token":self.jiushiToken
        }
        response = requests.post(url, data=json.dumps(body), headers=headers)
        print(response.status_code)
        print(response.json())

    def TellJiuShiRecovery(self):
        url = "https://gateway.zelostech.com.cn/business-server/open-apis/vehicle/command"
        body = {
            "vehicleName": "ZL01351",
            "commandType": "RECOVERY",
            "userId": "15",
            "userName": "testUser",
            "source": "34b5c696d54941e217938c590d45b598"
        }

        headers = {"Content-Type": "application/json", 
            "token":self.jiushiToken
        }
        response = requests.post(url, data=json.dumps(body), headers=headers)
        print(response.status_code)
        print(response.json())

    def TellJiuShiSound(self):
        #randInt = random.randrange(0, 5)
        #randSound = Sounds[randInt]
        randSound = "请选择货品"
        url = "https://gateway.zelostech.com.cn/business-server/open-apis/vehicle/sound_and_show"
        body = {
            "vehicleName": "ZL01351",
            "sound": randSound,
            "show": "买买买",
            "showDuration": "10"
        }

        headers = {"Content-Type": "application/json", 
            "token":self.jiushiToken
        }
        response = requests.post(url, data=json.dumps(body), headers=headers)
        print(response.status_code)
        print(response.json())

    def TellJiuShiInitCommandSound(self):
        sound = "图像识别已开启"
        url = "https://gateway.zelostech.com.cn/business-server/open-apis/vehicle/sound_and_show"
        body = {
            "vehicleName": "ZL01351",
            "sound": sound,
            "show": "检测开启",
            "showDuration": "3"
        }

        headers = {"Content-Type": "application/json", 
            "token":self.jiushiToken
        }
        response = requests.post(url, data=json.dumps(body), headers=headers)
        print(response.status_code)
        print(response.json())

    def TellJiuShiCommandSound(self):
        randInt = random.randrange(0, 3)
        randSound = CommandSounds[randInt]
        url = "https://gateway.zelostech.com.cn/business-server/open-apis/vehicle/sound_and_show"
        body = {
            "vehicleName": "ZL01351",
            "sound": randSound,
            "show": "买买买",
            "showDuration": "10"
        }

        headers = {"Content-Type": "application/json", 
            "token":self.jiushiToken
        }
        response = requests.post(url, data=json.dumps(body), headers=headers)
        print(response.status_code)
        print(response.json())

    def TellJiuShiCommandStopSide(self):
        url = "https://gateway.zelostech.com.cn/business-server/open-apis/vehicle/command"
        body = {
            "vehicleName": "ZL01351",
            "commandType": "BUSINESS_ONE_KEY_SIDE",
            "userId": "15",
            "userName": "testUser",
            "source": "34b5c696d54941e217938c590d45b598"
        }

        headers = {"Content-Type": "application/json", 
            "token":self.jiushiToken
        }
        response = requests.post(url, data=json.dumps(body), headers=headers)
        print(response.status_code)
        print(response.json())

    def TellJiuShiCommandRecoverySide(self):
        url = "https://gateway.zelostech.com.cn/business-server/open-apis/vehicle/command"
        body = {
            "vehicleName": "ZL01351",
            "commandType": "BUSINESS_TASK_RECOVERY",
            "userId": "15",
            "userName": "testUser",
            "source": "34b5c696d54941e217938c590d45b598"
        }

        headers = {"Content-Type": "application/json", 
            "token":self.jiushiToken
        }
        response = requests.post(url, data=json.dumps(body), headers=headers)
        print(response.status_code)
        print(response.json())

    def update(self):
        final_result = []
        norm_type = self.cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.cfg.DATA_PRESET.HEATMAP_SIZE
        if self.save_video:
            # initialize the file video stream, adapt ouput video resolution to original video
            stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            if not stream.isOpened():
                print("Try to use other video encoders...")
                ext = self.video_save_opt['savepath'].split('.')[-1]
                fourcc, _ext = self.recognize_video_ext(ext)
                self.video_save_opt['fourcc'] = fourcc
                self.video_save_opt['savepath'] = self.video_save_opt['savepath'][:-4] + _ext
                stream = cv2.VideoWriter(*[self.video_save_opt[k] for k in ['savepath', 'fourcc', 'fps', 'frameSize']])
            assert stream.isOpened(), 'Cannot open video for writing'
        # keep looping infinitelyd
        self.GetJiuShiToken()
        self.TellJiuShiInitCommandSound()
        while True:
            #self.GetJiuShiToken()

            curTime = time.time()
            if self.isStopping:
                if self.lastRecordTime != 0:
                    deltaTime = curTime - self.lastRecordTime
                    self.accStopTime += deltaTime
                    self.lastRecordTime = curTime

                    if self.accStopTime >= JiuShiStopTime:
                        # which means it has excceeded max waiting time
                        print("trigger recovery")
                        self.TellJiuShiCommandRecoverySide()
                        self.accStopTime = 0.0
                        self.accCommonSoundTime = 0.0
                        self.isStopping = False
                self.lastRecordTime = curTime
            else:
                if self.lastRecordTime != 0:
                    deltaTime = curTime - self.lastRecordTime
                    self.accCommonSoundTime += deltaTime
                    self.lastRecordTime = curTime

                    if self.accCommonSoundTime >= JiuShiCommonSoundTime:
                        # which means it has excceeded max waiting time
                        print("trigger command sound")
                        self.TellJiuShiCommandSound()
                        self.accCommonSoundTime = 0.0
                self.lastRecordTime = curTime

            if self.stopSignal:
                if self.isStopping == False:
                    print("trigger stop")
                    self.TellJiuShiCommandStopSide()
                    self.TellJiuShiSound()
                self.isStopping = True
                self.accStopTime = 0.0
                self.stopSignal = False
                self.accCommonSoundTime = 0.0

            # ensure the queue is not empty and get item
            (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, camera_idx) = self.wait_and_get(self.result_queue)
            if orig_img is None:
                # if the thread indicator variable is set (img is None), stop the thread
                if self.save_video:
                    stream.release()
                write_json(final_result, self.opt.outputpath, form=self.opt.format, for_eval=self.opt.eval)
                print("Results have been written to json.")
                return
            # image channel RGB->BGR
            orig_img = np.array(orig_img, dtype=np.uint8)[:, :, ::-1]
            if boxes is None or len(boxes) == 0:
                if self.opt.save_img or self.save_video or self.opt.vis:
                    self.write_image(orig_img, im_name, stream=stream if self.save_video else None)
            else:
                # location prediction (n, kp, 2) | score prediction (n, kp, 1)
                assert hm_data.dim() == 4

                face_hand_num = 110
                if hm_data.size()[1] == 136:
                    self.eval_joints = [*range(0,136)]
                elif hm_data.size()[1] == 26:
                    self.eval_joints = [*range(0,26)]
                elif hm_data.size()[1] == 133:
                    self.eval_joints = [*range(0,133)]
                elif hm_data.size()[1] == 68:
                    face_hand_num = 42
                    self.eval_joints = [*range(0,68)]
                elif hm_data.size()[1] == 21:
                    self.eval_joints = [*range(0,21)]
                pose_coords = []
                pose_scores = []
                for i in range(hm_data.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    if isinstance(self.heatmap_to_coord, list):
                        pose_coords_body_foot, pose_scores_body_foot = self.heatmap_to_coord[0](
                            hm_data[i][self.eval_joints[:-face_hand_num]], bbox, hm_shape=hm_size, norm_type=norm_type)
                        pose_coords_face_hand, pose_scores_face_hand = self.heatmap_to_coord[1](
                            hm_data[i][self.eval_joints[-face_hand_num:]], bbox, hm_shape=hm_size, norm_type=norm_type)
                        pose_coord = np.concatenate((pose_coords_body_foot, pose_coords_face_hand), axis=0)
                        pose_score = np.concatenate((pose_scores_body_foot, pose_scores_face_hand), axis=0)
                    else:
                        pose_coord, pose_score = self.heatmap_to_coord(hm_data[i][self.eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)
                if not self.opt.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = \
                        pose_nms(boxes, scores, ids, preds_img, preds_scores, self.opt.min_box_area, use_heatmap_loss=self.use_heatmap_loss)

                _result = []
                for k in range(len(scores)):
                    if (camera_idx == 0 or camera_idx == 1) and len(preds_scores[k]) > 18:
                        conf0 = preds_scores[k][0]
                        conf1 = preds_scores[k][1]
                        conf2 = preds_scores[k][2]
                        conf3 = preds_scores[k][3]
                        conf4 = preds_scores[k][4]

                        conf5 = preds_scores[k][5]
                        conf7 = preds_scores[k][7]
                        conf9 = preds_scores[k][9]

                        conf6 = preds_scores[k][6]
                        conf8 = preds_scores[k][8]
                        conf10 = preds_scores[k][10]

                        conf18 = preds_scores[k][18]


                        x0, y0 = preds_img[k][0]

                        x1, y1 = preds_img[k][1]
                        x2, y2 = preds_img[k][2]
                        x3, y3 = preds_img[k][3]
                        x4, y4 = preds_img[k][4]

                        x5, y5 = preds_img[k][5]
                        x7, y7 = preds_img[k][7]
                        x9, y9 = preds_img[k][9]

                        x6, y6 = preds_img[k][6]
                        x8, y8 = preds_img[k][8]
                        x10, y10 = preds_img[k][10]

                        x18, y18 = preds_img[k][18]

                        if conf0 > 0.4 and conf1 > 0.4 and conf2 > 0.4 and conf3 > 0.4 and conf4 > 0.4 and conf18 > 0.4:
                            v018 = [x18 - x0, y18 - y0]
                            v12 = [x1 - x2, y1 - y2]
                            v13 = [x1 - x3, y1 - y3]
                            v24 = [x2 - x4, y2 - y4]
                            len018 = np.linalg.norm(v018)
                            len12 = np.linalg.norm(v12)
                            len13 = np.linalg.norm(v13)
                            len24 = np.linalg.norm(v24)
                            if len018 > 65:
                                #print("which means some is closed")
                                self.stopSignal = True

                            # len018 to tell global distance
                            # len12 to tell partial facial direction
                            if len018 > 25 and len018 / len12 < 2:
                                ratio = len018 / len12
                                #print("someone is closed and look: " + str(ratio))
                                self.stopSignal = True

                        if (conf5 > 0.4 and conf7 > 0.4):
                            v75 = [x7 - x5, y7 - y5]
                            len75 = np.linalg.norm(v75)
                            if len75 > 25 and y9 < y5:
                                #print("someone is waving left hand")
                                self.stopSignal = True

                        if (conf6 > 0.4 and conf8 > 0.4):
                            v86 = [x8 - x6, y8 - y6]
                            len86 = np.linalg.norm(v86)
                            if len86 > 25 and y10 < y6:
                                #print("someone is waving right hand")
                                self.stopSignal = True


                    _result.append(
                        {
                            'keypoints':preds_img[k],
                            'kp_score':preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            'idx':ids[k],
                            'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]] 
                        }
                    )

                result = {
                    'imgname': im_name,
                    'result': _result
                }

                from alphapose.utils.vis import vis_frame_fast as vis_frame
                testimg = vis_frame(orig_img, result, self.opt, self.vis_thres)
                imgname = "kkk" + str(camera_idx) + ".jpg"
                cv2.imwrite(imgname, testimg)


                if self.opt.pose_flow:
                    poseflow_result = self.pose_flow_wrapper.step(orig_img, result)
                    for i in range(len(poseflow_result)):
                        result['result'][i]['idx'] = poseflow_result[i]['idx']

                final_result.append(result)
                if self.opt.save_img or self.save_video or self.opt.vis:
                    if hm_data.size()[1] == 49:
                        from alphapose.utils.vis import vis_frame_dense as vis_frame
                    elif self.opt.vis_fast:
                        from alphapose.utils.vis import vis_frame_fast as vis_frame
                    else:
                        from alphapose.utils.vis import vis_frame
                    img = vis_frame(orig_img, result, self.opt, self.vis_thres)
                    self.write_image(img, im_name, stream=stream if self.save_video else None)

    def write_image(self, img, im_name, stream=None):
        if self.opt.vis:
            cv2.imshow("AlphaPose Demo" + str(self.camera_idx), img)
            cv2.waitKey(30)
        if self.opt.save_img:
            cv2.imwrite(os.path.join(self.opt.outputpath, 'vis', im_name), img)
        if self.save_video:
            stream.write(img)

    def wait_and_put(self, queue, item):
        queue.put(item)

    def wait_and_get(self, queue):
        return queue.get()

    def save(self, boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, camera_idx = 0):
        # save next frame in the queue
        self.wait_and_put(self.result_queue, (boxes, scores, ids, hm_data, cropped_boxes, orig_img, im_name, camera_idx))

    def running(self):
        # indicate that the thread is still running
        return not self.result_queue.empty()

    def count(self):
        # indicate the remaining images
        return self.result_queue.qsize()

    def stop(self):
        # indicate that the thread should be stopped
        self.save(None, None, None, None, None, None, None)
        self.result_worker.join()

    def terminate(self):
        # directly terminate
        self.result_worker.terminate()

    def clear_queues(self):
        self.clear(self.result_queue)
        
    def clear(self, queue):
        while not queue.empty():
            queue.get()

    def results(self):
        # return final result
        print(self.final_result)
        return self.final_result

    def recognize_video_ext(self, ext=''):
        if ext == 'mp4':
            return cv2.VideoWriter_fourcc(*'mp4v'), '.' + ext
        elif ext == 'avi':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        elif ext == 'mov':
            return cv2.VideoWriter_fourcc(*'XVID'), '.' + ext
        else:
            print("Unknow video format {}, will use .mp4 instead of it".format(ext))
            return cv2.VideoWriter_fourcc(*'mp4v'), '.mp4'

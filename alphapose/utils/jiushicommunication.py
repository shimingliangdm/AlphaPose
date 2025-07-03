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

class jiushicommunication():
    def __init__(self, ):
        self.curJiuShiTokenTime = JiuShiTokenExpireTime
        self.jiushiToken = ""
        self.lastRecordTime = 0
        self.isStopping = False
        self.accStopTime = 0.0
        self.accCommonSoundTime = 0.0
        self.stopSignal = False

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

    
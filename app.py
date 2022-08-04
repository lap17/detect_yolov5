import sys
import gc
from typing import List
from streamlit_webrtc import ClientSettings
from typing import List, NamedTuple, Optional
import cv2
import base64
import torch
import numpy as np
import pandas as pd
import streamlit as st
import pytz
import av
import datetime
import matplotlib.colors as mcolors
from PIL import Image
import time
import threading
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer, WebRtcMode, RTCConfiguration
from streamlit.legacy_caching import clear_cache
from aiortc.contrib.media import MediaPlayer

lock = threading.Lock()

DEFAULT_CONFIDENCE_THRESHOLD = 0.4

CLASSES_CUSTOM = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','vest','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress','vest dress','sling dress']

def main():
    gc.enable()
    st.header("Fashion Items Detection Demo")
    st.sidebar.markdown("""<center data-parsed=""><img src="http://drive.google.com/uc?export=view&id=1D-pN81CupHMcxb7xa5-Z6JZZIbagRqH_" align="center"></center>""",unsafe_allow_html=True,)
    st.sidebar.markdown(" ")
    
    def reload():
        clear_cache()
        gc.collect()
        st.experimental_rerun()
        
    pages = st.sidebar.columns([1, 1, 1])
    pages[0].markdown(" ")
    
    if pages[1].button("Reload App"):
        reload()
    prediction_mode = st.sidebar.radio("", ('Single image', 'Web camera', 'Local video'), index=2)
    if prediction_mode == 'Single image':
        pass
    elif prediction_mode == 'Web camera':
        pass
    elif prediction_mode == 'Local video':
        func_video()
        
def get_yolo5():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='last_s.pt')

def get_preds(img):
    return model([img]).xyxy[0].numpy()

def get_colors(indexes):
    to_255 = lambda c: int(c*255)
    tab_colors = list(mcolors.TABLEAU_COLORS.values())
    tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) for name_color in tab_colors]
    base_colors = list(mcolors.BASE_COLORS.values())
    base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
    rgb_colors = tab_colors + base_colors
    rgb_colors = rgb_colors*5
    color_dict = {}
    for i, index in enumerate(indexes):
        if i < len(rgb_colors):
            color_dict[index] = rgb_colors[i]
        else:
            color_dict[index] = (255,0,0)
    return color_dict
    
def create_player(path):
    return MediaPlayer(path)


def transform(frame):
    img = frame.to_ndarray(format="bgr24")
    img_ch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = get_preds(img_ch)
    result = result[np.isin(result[:,-1], target_class_ids)]  
    for bbox_data in result:
        xmin, ymin, xmax, ymax, conf, label = bbox_data
        if conf > confidence_threshold:
            xmin = int(xmin)
            ymin = int(ymin)
            xmax = int(xmax)
            ymax = int(ymax)
            p0, p1, label = (xmin, ymin), (xmax, ymax), int(label)
            img = cv2.rectangle(img, p0, p1, rgb_colors[label], 2) 
            ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
            xtext = xmin + 10
            class_ = CLASSES[label]
            text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
            img = cv2.putText(img, text_for_vis, (int(xtext), int(ytext)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[label], 2,)
            if agree:
                time_detect = datetime.datetime.now(pytz.timezone("America/New_York")).replace(tzinfo=None).strftime("%m-%d-%y %H:%M:%S")
                cropped_image = img[ymin:ymax, xmin:xmax]
                retval, buffer_img= cv2.imencode('.jpg', cropped_image)
                data = base64.b64encode(buffer_img).decode("utf-8")
                html = "<img src='data:image/jpg;base64," + data + f"""' style='display:block;margin-left:auto;margin-right:auto;width:200px;border:0;'>"""
                with lock:
                    result_queue.insert(0, {'object': class_, 'time_detect': time_detect, 'confident': str(conf.round(2)), 'img': html})
    return av.VideoFrame.from_ndarray(img, format="bgr24")
        
def func_image():
    pass
    
    
def func_web():
    pass
    
    
def func_video():
    RTC_CONFIGURATION = RTCConfiguration(
       {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    ) 

    CLASSES_CUSTOM = [ 'short sleeve top', 'long sleeve top','short sleeve outwear','long sleeve outwear','vest','sling','shorts','trousers','skirt','short sleeve dress', 'long sleeve dress','vest dress','sling dress']
    

    DEFAULT_CONFIDENCE_THRESHOLD = 0.4

    result_queue = [] 

    def get_yolo5():
        return torch.hub.load('ultralytics/yolov5', 'custom', path='last_s.pt')

    def get_preds(img):
        return model([img]).xyxy[0].numpy()

    def get_colors(indexes):
        to_255 = lambda c: int(c*255)
        tab_colors = list(mcolors.TABLEAU_COLORS.values())
        tab_colors = [list(map(to_255, mcolors.to_rgb(name_color))) for name_color in tab_colors]
        base_colors = list(mcolors.BASE_COLORS.values())
        base_colors = [list(map(to_255, name_color)) for name_color in base_colors]
        rgb_colors = tab_colors + base_colors
        rgb_colors = rgb_colors*5
        color_dict = {}
        for i, index in enumerate(indexes):
            if i < len(rgb_colors):
                color_dict[index] = rgb_colors[i]
            else:
                color_dict[index] = (255,0,0)
        return color_dict
    
    def create_player():
        return MediaPlayer('test.mp4')


    def transform(frame):
        img = frame.to_ndarray(format="bgr24")
        img_ch = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = get_preds(img_ch)
        result = result[np.isin(result[:,-1], target_class_ids)]  
        for bbox_data in result:
            xmin, ymin, xmax, ymax, conf, label = bbox_data
            if conf > confidence_threshold:
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                p0, p1, label = (xmin, ymin), (xmax, ymax), int(label)
                img = cv2.rectangle(img, p0, p1, rgb_colors[label], 2) 
                ytext = ymin - 10 if ymin - 10 > 10 else ymin + 15
                xtext = xmin + 10
                class_ = CLASSES[label]
                text_for_vis = '{} {}'.format(class_, str(conf.round(2)))
                img = cv2.putText(img, text_for_vis, (int(xtext), int(ytext)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rgb_colors[label], 2,)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
    




    with st.spinner('Loading the model...'):
        cache_key = 'custom'
        if cache_key in st.session_state:
            model = st.session_state[cache_key]
        else:
            model = get_yolo5()
            st.session_state[cache_key] = model
            
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, DEFAULT_CONFIDENCE_THRESHOLD, 0.05)

    #prediction_mode = st.sidebar.radio("", ('Single image', 'Web camera', 'Local video'), index=2)
    CLASSES = CLASSES_CUSTOM
    classes_selector = st.sidebar.multiselect('Select classes', CLASSES, default='short sleeve top')

    all_labels_chbox = st.sidebar.checkbox('All classes', value=True)
    if all_labels_chbox:
        target_class_ids = list(range(len(CLASSES)))
    elif classes_selector:
        target_class_ids = [CLASSES.index(class_name) for class_name in classes_selector]
    else:
        target_class_ids = [0]
    rgb_colors = get_colors(target_class_ids)
    detected_ids = None
    #player = create_player("test.mp4")
    ctx_l = webrtc_streamer(
            key="key",
            mode=WebRtcMode.RECVONLY,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={
                "video": True,
                "audio": False,
            },
            player_factory=create_player,
            video_frame_callback=transform
        )

if __name__ == "__main__":
    main()
from ultralytics import YOLO
import cv2
import torch
from tracker import *
import numpy as np
import pandas as pd
from ultralytics.solutions import object_counter
#model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True) #Prebuild Yolo5s model
model=YOLO("yolov8n.pt")



cap=cv2.VideoCapture("C:/Users/annaa/Downloads/crowd.mp4") # input video



#saving output video
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('output_video.avi',fourcc,20.0,(1020,500))



#function to play sound when count hits threshold
import simpleaudio as sa
def play_sound(sound_file):
    try:
        print(f"Trying to play: {sound_file}")
        wave_obj = sa.WaveObject.from_wave_file(sound_file)
        play_obj = wave_obj.play()
        play_obj.wait_done()
    except Exception as e:
        print(f"Error playing sound: {e}")

sound_file = "C:/Users/annaa/Downloads/alert1.wav"





#To find the cordinates for...Oru particular area edth athil entry exit check cheyth count edukkan..

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN :  
        colorsBGR = [x, y]
        print(colorsBGR)
    
points = []

def draw_polyline(event, x, y, flags, param):
    global points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 5: # 5 point vare ulla polygon create cheyyan
            points.append((x, y))
            print(f"Point {len(points)} added: ({x}, {y})")
        #if len(points) == 4:
        #    cv2.polylines(frame, [np.array(points, np.int32)], True, (255, 255, 255), 2)
        #cv2.imshow('Frame', frame)


cv2.namedWindow('FRAME')
#cv2.setMouseCallback('FRAME', POINTS)
cv2.setMouseCallback('FRAME', draw_polyline)




my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")





tracker = Tracker()

#ROI=[(74, 471),(171, 184),(593, 179),(799, 435)]
ROI=[(0, 0),(0, 0),(0, 0),(0,0)]
#classes_to_count = [0]
area=set()

#Ultralytics provide cheyyunna counter module..Ith use cheyth manual ROI set cheyyaam..
#counter = object_counter.ObjectCounter()
#counter.set_args(view_img=True,
#                 reg_pts=ROI,
#                 classes_names=model.names)

#print(counter.reg_pts)






while True:
    ret,frame=cap.read()
    
    frame=cv2.resize(frame,(1020,500))

    #print("Region Points :: {}".format(counter.reg_pts))
    #print("ROI :: {}".format(ROI))
    #ROI = counter.reg_pts
    
    print(points)
    #Ivide 4 points nml select cheyyunnu...
    #Aaa pointsil minimum 3 points undenkil polylines varachaal mathi..
    
    
    if len(points) > 2:
        ROI = points
    cv2.polylines(frame,
        [np.array(ROI,np.int32)],
        True,
        (255,255,255),3)
    
    
    #tracks = model.track(frame, persist=True, show=False,
    #                     classes=0)
    #frame0 = counter.start_counting(frame, tracks)

    
    passframe=model.predict(frame)
    
    
    a=passframe[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    
    list_for_coords = []
    
    for index,row in px.iterrows():
        
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5]) # Ith 4 or 5 aano enn check cheyyth change cheyynm.. Jupiter notbook vach chelpol 5th element aayitt aavm return cheyyuka..
        
        c=class_list[d]
        if 'person' in c:
            list_for_coords.append([x1,y1,x2,y2])
            
    object_ids = tracker.update(list_for_coords)
    #print(object_ids) # Cordinatesum IDyum list cheyth kittum..

    for object_id in object_ids:
        x,y,w,h,id = object_id
        cv2.rectangle(frame,
            (x,y),(w,h),
            (255,255,255),2)
        
        cv2.putText(frame,
            str(id),(w,h),
            cv2.FONT_ITALIC,0.5,
            (0,0,255),2)

         #Point Polygon Test
        result = cv2.pointPolygonTest(
            np.array(ROI,np.int32),
            (int((x+w)/2), int((y+h)/2)),    #(w,y)
            False)
        #print("RESULTS {}".format(result))
        
        
        
        if result>0:
            area.add(id)
    count=len(area)
    print(count)
    
    
    cv2.putText(frame, f'People Count: {count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('People Counting', frame)
    
    
    
    
    
    #Trigger an alarm if the count exceeds the threshold.
    
    if count > 8:
        cv2.putText(frame, 'Crowded!', (438, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        # Ith maathram threshold hit aayaal aa if statementinte ullil kodukkanm...Aa crowded enn print cheyyunnidath...
        play_sound(sound_file)

    
    
    area=set()
    
    
    out.write(frame)
    
    
    cv2.imshow('FRAME',frame)   
    if cv2.waitKey(1)& 0xFF == ord('q'):
        break
       

cap.release()
out.release()
cv2.destroyAllWindows()








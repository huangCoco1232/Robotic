# Robotic challenge of Leiden university 2025 spring
Robotic base on the raspberry 4B

#Welcome to the robotic challenge:
#final.py is the main function contral the project:
  if you want run this project,please replace the YOLOv5_lite model path to your own path:
  and the download [link](https://github.com/ppogg/YOLOv5-Lite)
  here:"sys.path.insert(0, "/media/pi/robo/v5lite/YOLOv5-Lite")"
  and "ckpt = torch.load("/media/pi/robo/v5lite/YOLOv5-Lite/v5lite-e.pt", map_location="cpu")"

#run the project:
into raspberry os
>>python final.py 

#to save the BEV graph:
press "s"
#to stop the robot:
press "q"

#to stop the project:
>>ctrl+c to stop

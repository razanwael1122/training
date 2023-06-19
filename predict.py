from ultralytics  import YOLO
model = YOLO("best.onnx")
model.predict(source="1 (6).png" ,show=True,save=True,show_labels=True,show_conf=True,conf=0.5,save_txt=False,
              save_crop=False,line_thickness=2,box=True)
from ultralytics import YOLO
import cv2
# Load a model

# model = YOLO('yolov8n-seg.pt')
model = YOLO("yolov8n.pt")  # load a pretrained model (recsommended for training)
video_path = "C:/Users/Peace/workspace/dataset/mp4/1.mp4"
cap = cv2.VideoCapture(0)
while cap.isOpened():
    status, frame = cap.read()
    if not status:
        break
    results = model.predict(source = frame)
    result = results[0]     
    anno_frame = result.plot()
    cv2.imshow("V8",anno_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows

# Use the model




# save_path = "D:\workspace\mp4"
# results = model.predict(source, stream=True)
# print(results)

# model = YOLO("yolov8n.pt")

# model = YOLO('yolov8n-seg.pt')
# results = model("C:/Users/Peace/workspace/dataset/1.png",save=True)
# source = 'C:/Users/Peace/workspace/dataset/1.jpg'

# # source = 'C:/Users/Peace/workspace/dataset/1.jpg'
# # model.predict(source, save=True)

# results = model(source) 







# # 调取摄像头
# cap = cv2.VideoCapture(0)  # 参数0表示第一个摄像头

# # 检查摄像头是否成功开启
# if not cap.isOpened():
#     print("无法开启摄像头")
#     exit()

# try:
#     while True:
#         # 捕获摄像头的一帧
#         ret, frame = cap.read()

#         # 如果正确读取帧，ret为True
#         if not ret:
#             print("无法从摄像头读取帧")
#             break

#         # 显示结果帧
#         cv2.imshow('Camera Frame', frame)

#         # 按 'q' 键退出
#         if cv2.waitKey(1) == ord('q'):
#             break
# finally:
#     # 释放摄像头资源
#     cap.release()
#     # 关闭所有活动的窗口
#     cv2.destroyAllWindows()



# image_path = 'C:/Users/Peace/workspace/dataset/01.jpg'
# cv2.imread
# image = cv2.imread(image_path)
# if image is not None:
#     # 使用 cv2.imshow() 显示图片
#     cv2.imshow('Image', image)

#     # 等待按键后关闭窗口
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()
# else:
#     print("Error: 图片无法加载，请检查路径。")

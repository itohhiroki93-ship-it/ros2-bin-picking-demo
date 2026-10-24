#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import cv2
from cv_bridge import CvBridge

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')
        
        # 1. ROS 2パラメータの宣言 (初期値の設定)
        self.declare_parameter('camera_index', -1)  # -1 の場合は外付けカメラを自動検出
        self.declare_parameter('frame_width', 640)
        self.declare_parameter('frame_height', 480)
        self.declare_parameter('fps', 15.0)         # ArUcoマーカーやYOLOの負荷を考慮して15Hzをデフォルト値に設定

        # パラメータの取得
        self.camera_index = self.get_parameter('camera_index').get_parameter_value().integer_value
        self.frame_width = self.get_parameter('frame_width').get_parameter_value().integer_value
        self.frame_height = self.get_parameter('frame_height').get_parameter_value().integer_value
        self.fps = self.get_parameter('fps').get_parameter_value().double_value

        self.bridge = CvBridge()
        self.publisher_ = self.create_publisher(Image, 'camera/image_raw', 10)
        
        # 2. カメラデバイスの探索と初期化
        self.init_camera()

        # 3. 画像配信用タイマーの作成
        timer_period = 1.0 / self.fps
        self.timer = self.create_timer(timer_period, self.timer_callback)
        self.get_logger().info(f"カメラノードが起動しました。配信周期: {self.fps} Hz (周期: {timer_period:.3f} 秒)")

    def find_external_usb_camera(self):
        """
        ジェミニ参照の inference.py にあるロジックに基づき、
        PC内蔵カメラ (integrated/internal等) を自動的に除外して、
        Pixel 6aなどの外付けカメラデバイスを自動探索する関数。
        """
        if not os.path.exists("/sys/class/video4linux"):
            # video4linuxが存在しない場合は 0〜9 番を順にテスト
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    ret, _ = cap.read()
                    cap.release()
                    if ret: 
                        return i, f"Camera {i}"
            return -1, ""

        integrated_keywords = ["integrated", "internal", "builtin", "laptop", "webcam0", "pc cam", "chicony", "sunplus"]
        external_candidates = []
        
        for dev in sorted(os.listdir("/sys/class/video4linux")):
            if not dev.startswith("video"): 
                continue
            try:
                idx = int(dev.replace("video", ""))
                dev_path = os.path.join("/sys/class/video4linux", dev)
                name_file = os.path.join(dev_path, "name")
                cam_name = f"video{idx}"
                
                if os.path.exists(name_file):
                    with open(name_file, "r", encoding="utf-8", errors="ignore") as f:
                        cam_name = f.read().strip()

                # 内蔵カメラに該当する名前であればスキップ
                if any(kw in cam_name.lower() for kw in integrated_keywords):
                    continue

                # 実際にオープンして画像を取得できるか確認
                cap_test = cv2.VideoCapture(idx)
                if cap_test.isOpened():
                    ret, _ = cap_test.read()
                    cap_test.release()
                    if ret: 
                        external_candidates.append((idx, cam_name))
            except Exception: 
                continue

        if len(external_candidates) > 0:
            return external_candidates[0][0], external_candidates[0][1]
        
        return -1, ""

    def init_camera(self):
        # インデックスが-1の場合は、外付けカメラの自動探索を行う
        if self.camera_index == -1:
            idx, cam_name = self.find_external_usb_camera()
            if idx >= 0:
                self.get_logger().info(f"外付けカメラを自動検出しました: {cam_name} (/dev/video{idx})")
                self.camera_index = idx
            else:
                self.get_logger().warn("外付けカメラが見つかりません。デフォルトのビデオインデックス 0 を使用します。")
                self.camera_index = 0

        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            self.get_logger().error(f"カメラ (デバイス番号: {self.camera_index}) の初期化に失敗しました。")
            self.use_dummy = True
            self.get_logger().warn("テスト用のダミー画像配信モードで動作します。")
            return
        
        self.use_dummy = False
        
        # 解像度の設定
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.get_logger().info(f"カメラ解像度を設定しました: {actual_w}x{actual_h}")

    def timer_callback(self):
        if self.use_dummy:
            # カメラが接続されていない場合のダミー画像生成 (赤い円が動き回る)
            import numpy as np
            frame = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)
            t = self.get_clock().now().nanoseconds / 1e9
            cx = int(self.frame_width / 2 + (self.frame_width / 4) * np.sin(t))
            cy = int(self.frame_height / 2 + (self.frame_height / 4) * np.cos(t))
            cv2.circle(frame, (cx, cy), 20, (0, 0, 255), -1)
            cv2.putText(frame, "PHYSICAL CAMERA NOT CONNECTED (DUMMY)", (30, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.get_logger().error("ビデオフレームのキャプチャに失敗しました。")
                return

        # OpenCV(BGR)の画像フレームをROS 2のImageメッセージに変換して送信
        try:
            msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            self.publisher_.publish(msg)
        except Exception as e:
            self.get_logger().error(f"画像の配信（Publish）中にエラーが発生しました: {e}")

    def destroy_node(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
            self.get_logger().info("カメラデバイスを解放しました。")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
import os
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
import cv2
from cv_bridge import CvBridge
import numpy as np

# YOLO (ultralytics) のインポート
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ObjectDetectionNode(Node):
    def __init__(self):
        super().__init__('object_detection_node')
        
        # 1. パラメータの宣言
        default_model_path = os.path.join(
            '/home/mechatro/ros2_ws/src/magician_ros2/dobot_itou01/dobot_itou01/ジェミニ参照',
            'best.pt'
        )
        self.declare_parameter('model_path', default_model_path)
        self.declare_parameter('conf_threshold', 0.5)
        
        # ユーザー指定の画像変換（ROI）パラメータ
        self.declare_parameter('zoom', 2.02)
        self.declare_parameter('crop_left', 0.27)
        self.declare_parameter('crop_right', 0.01)
        self.declare_parameter('crop_top', 0.00)
        self.declare_parameter('crop_bottom', 0.22)
        
        self.declare_parameter('publish_debug_image', True)

        # パラメータの取得
        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.conf_threshold = self.get_parameter('conf_threshold').get_parameter_value().double_value
        
        self.zoom = self.get_parameter('zoom').get_parameter_value().double_value
        self.crop_left = self.get_parameter('crop_left').get_parameter_value().double_value
        self.crop_right = self.get_parameter('crop_right').get_parameter_value().double_value
        self.crop_top = self.get_parameter('crop_top').get_parameter_value().double_value
        self.crop_bottom = self.get_parameter('crop_bottom').get_parameter_value().double_value
        
        self.publish_debug_image = self.get_parameter('publish_debug_image').get_parameter_value().bool_value

        self.bridge = CvBridge()
        
        # アームの動作状態フラグ (初期値は安全のため動作中=True)
        self.arm_moving = True

        # 2. YOLOモデルの読み込み
        global YOLO_AVAILABLE
        if YOLO_AVAILABLE:
            if os.path.exists(self.model_path):
                self.get_logger().info(f"YOLOモデルをロードしています: {self.model_path}")
                try:
                    self.model = YOLO(self.model_path)
                    self.get_logger().info("YOLOモデルのロードに成功しました。")
                except Exception as e:
                    self.get_logger().error(f"YOLOモデルのロードに失敗しました: {e}")
                    YOLO_AVAILABLE = False
            else:
                self.get_logger().error(f"モデルファイルが存在しません: {self.model_path}")
                YOLO_AVAILABLE = False
        else:
            self.get_logger().error("ultralytics (YOLO) パッケージがインポートできません。ダミー検出モードで動作します。")
            
        # 3. パブリッシャとサブスクライバの作成
        self.pixel_pub = self.create_publisher(Point, 'detected_object_pixel', 10)
        self.debug_image_pub = self.create_publisher(Image, 'camera/detected_image', 10)
        
        self.arm_status_sub = self.create_subscription(
            Bool,
            'arm_status',
            self.arm_status_callback,
            10
        )
        
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )
        
        self.get_logger().info("物体検出ノードが初期化されました。")

    def arm_status_callback(self, msg):
        self.arm_moving = msg.data
        status_str = "動作中 (BUSY)" if self.arm_moving else "停止中 (READY)"
        self.get_logger().info(f"アームの動作状況を更新しました: {status_str}")

    def send_debug_image(self, frame, header):
        """
        🌟 完全無欠のデバッグ画像配信関数
        CvBridgeのバグを回避し、メモリの再構築で「白黒の色化け」を100%防ぎます。
        """
        if not self.publish_debug_image:
            return

        try:
            from sensor_msgs.msg import Image as ROSImage
            import numpy as np
            
            # 🌟 ここが超重要！メモリ上の見えない隙間（パディング）を排除し、
            # 完全に連続した綺麗なピクセル配列として新品のキャンバスに再構築します。
            clean_frame = np.ascontiguousarray(frame.copy())
            h, w = clean_frame.shape[:2]

            # CvBridgeを使わず、ROSが一番喜ぶピュアな形式で直接メッセージを作ります
            debug_msg = ROSImage()
            debug_msg.header = header
            debug_msg.height = h
            debug_msg.width = w
            debug_msg.encoding = "bgr8"
            debug_msg.is_bigendian = 0
            debug_msg.step = w * 3  # 1行あたりの正確なバイト数（3チャンネル）
            
            # クリーンなメモリデータをそのまま流し込む
            debug_msg.data = clean_frame.tobytes()
            
            self.debug_image_pub.publish(debug_msg)
        except Exception as ex:
            self.get_logger().error(f"デバッグ画像配信エラー: {ex}")

    def apply_roi(self, frame):
        """
        指定パラメータに基づく画像切り抜きとズーム
        """
        h, w = frame.shape[:2]
        
        left = int(w * self.crop_left)
        right = int(w * (1.0 - self.crop_right))
        top = int(h * self.crop_top)
        bottom = int(h * (1.0 - self.crop_bottom))
        
        if left >= right or top >= bottom:
            left, right, top, bottom = 0, w, 0, h
            
        cropped = frame[top:bottom, left:right]
        
        if self.zoom > 1.0:
            ch, cw = cropped.shape[:2]
            new_w, new_h = int(cw / self.zoom), int(ch / self.zoom)
            start_x = (cw - new_w) // 2
            start_y = (ch - new_h) // 2
            cropped = cropped[start_y:start_y + new_h, start_x:start_x + new_w]
            
        return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    def roi_to_raw_coords(self, roi_x, roi_y, raw_w, raw_h):
        """
        ROI座標系(切り抜き・ズーム後)の座標を、
        ありのままのカメラ生映像(Raw Image)の座標系へ完全に逆算する関数
        """
        left = int(raw_w * self.crop_left)
        right = int(raw_w * (1.0 - self.crop_right))
        top = int(raw_h * self.crop_top)
        bottom = int(raw_h * (1.0 - self.crop_bottom))
        
        if left >= right or top >= bottom:
            left, right, top, bottom = 0, raw_w, 0, raw_h
            
        cw = right - left
        ch = bottom - top
        
        if self.zoom > 1.0:
            new_w, new_h = int(cw / self.zoom), int(ch / self.zoom)
            start_x = (cw - new_w) // 2
            start_y = (ch - new_h) // 2
        else:
            new_w, new_h = cw, ch
            start_x, start_y = 0, 0
            
        # 1. リサイズ(拡大)の逆算
        x_in_zoom = roi_x * (new_w / float(raw_w))
        y_in_zoom = roi_y * (new_h / float(raw_h))
        
        # 2. ズームオフセットの加算
        x_in_crop = x_in_zoom + start_x
        y_in_crop = y_in_zoom + start_y
        
        # 3. クロップ(切り抜き)オフセットの加算 -> これが生映像の絶対座標！
        raw_x = x_in_crop + left
        raw_y = y_in_crop + top
        
        return raw_x, raw_y

    def select_best_target(self, result, img_shape):
        if len(result.boxes) == 0:
            return None, None

        num_objects = len(result.boxes)
        img_h, img_w = img_shape[:2]
        center_x_img, center_y_img = img_w / 2.0, img_h / 2.0

        best_idx = -1
        best_score = -float('inf')
        best_center = None

        for i in range(num_objects):
            if result.masks is not None and len(result.masks) > i:
                mask_data = result.masks[i].data.cpu().numpy()
                area = np.sum(mask_data > 0)
            else:
                box_xywh = result.boxes.xywh[i].cpu().numpy()
                area = box_xywh[2] * box_xywh[3]

            conf = float(result.boxes.conf[i].item())
            box_x = float(result.boxes.xywh[i][0].item())
            box_y = float(result.boxes.xywh[i][1].item())
            dist = np.sqrt((box_x - center_x_img)**2 + (box_y - center_y_img)**2)

            score = (1.0 * area) + (5000.0 * conf) - (0.5 * dist)
            
            if score > best_score:
                best_score = score
                best_idx = i
                best_center = (box_x, box_y)

        if best_idx >= 0:
            return best_idx, best_center
        
        return None, None

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"画像変換エラー: {e}")
            return

        # 元のカメラ生画像のサイズを取得（逆算に使用）
        raw_h, raw_w = cv_image.shape[:2]

        # 1. ユーザー指定の変換処理 (ROI: カット＆ズーム) を適用
        processed_image = self.apply_roi(cv_image)

        # 2. アームが動作中の場合は、推論処理をスキップ
        if self.arm_moving:
            self.get_logger().debug("アームが動作中のため、物体検出（YOLO推論）をスキップします。")
            debug_frame = processed_image.copy()
            cv2.putText(debug_frame, "ARM BUSY - DETECTION PAUSED", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
            
            # 安全配信ヘルパーで送信！
            self.send_debug_image(debug_frame, msg.header)
            return

        # 3. 推論処理 (アーム停止中のみ実行)
        if YOLO_AVAILABLE:
            try:
                results = self.model.predict(source=processed_image, conf=self.conf_threshold, save=False, verbose=False)
                best_idx, best_center = self.select_best_target(results[0], processed_image.shape)
                
                if best_center is not None:
                    # 検出されたROI座標を、ありのままのカメラ生座標系 (Raw) へ逆算！
                    roi_x, roi_y = best_center[0], best_center[1]
                    raw_x, raw_y = self.roi_to_raw_coords(roi_x, roi_y, raw_w, raw_h)
                    
                    # 生座標をトピック配信
                    point_msg = Point()
                    point_msg.x = float(raw_x)
                    point_msg.y = float(raw_y)
                    point_msg.z = 0.0
                    self.pixel_pub.publish(point_msg)
                    # 🌟 ここに追加！
                    # 座標を送信した瞬間、ロボットからのBUSY信号を待たずに「自分自身で」推論をストップさせる！
                    self.arm_moving = True
                    
                    self.get_logger().info(
                        f"ターゲット検出 -> [生カメラ座標(Raw)]: X={raw_x:.1f}, Y={raw_y:.1f}  (ROI内: X={roi_x:.1f}, Y={roi_y:.1f})"
                    )
                    
                    if self.publish_debug_image:
                        debug_frame = processed_image.copy()
                        try:
                            box = results[0].boxes[best_idx].xyxy[0].cpu().numpy()
                            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                            cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        except Exception:
                            pass
                        
                        cv2.circle(debug_frame, (int(roi_x), int(roi_y)), 5, (0, 255, 0), -1)
                        cv2.putText(debug_frame, f"RAW TARGET ({raw_x:.1f}, {raw_y:.1f})", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        
                        # 🌟 最強の安全配信ヘルパーで送信！（これでグラデーションバグが消えます）
                        self.send_debug_image(debug_frame, msg.header)
                else:
                    if self.publish_debug_image:
                        debug_frame = processed_image.copy()
                        cv2.putText(debug_frame, f"NO OBJECT DETECTED (RAW MODE)", 
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # 安全配信ヘルパーで送信！
                        self.send_debug_image(debug_frame, msg.header)
                        
            except Exception as e:
                self.get_logger().error(f"YOLO推論中にエラーが発生しました: {e}")
        else:
            # ダミー検出モードでの処理
            hsv = cv2.cvtColor(processed_image, cv2.COLOR_BGR2HSV)
            lower_red1 = np.array([0, 70, 50])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([170, 70, 50])
            upper_red2 = np.array([180, 255, 255])
            
            mask = cv2.inRange(hsv, lower_red1, upper_red1) + cv2.inRange(hsv, lower_red2, upper_red2)
            M = cv2.moments(mask)
            
            if M["m00"] > 0:
                cX = float(M["m10"] / M["m00"])
                cY = float(M["m01"] / M["m00"])
                
                raw_h, raw_w = cv_image.shape[:2]
                raw_x, raw_y = self.roi_to_raw_coords(cX, cY, raw_w, raw_h)
                
                point_msg = Point()
                point_msg.x = float(raw_x)
                point_msg.y = float(raw_y)
                point_msg.z = 0.0
                self.pixel_pub.publish(point_msg)
                
                if self.publish_debug_image:
                    debug_frame = processed_image.copy()
                    cv2.circle(debug_frame, (int(cX), int(cY)), 10, (0, 255, 0), -1)
                    cv2.putText(debug_frame, f"DUMMY RAW ({raw_x:.1f}, {raw_y:.1f})", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    self.send_debug_image(debug_frame, msg.header)
            else:
                if self.publish_debug_image:
                    debug_frame = processed_image.copy()
                    cv2.putText(debug_frame, "NO RED OBJECT DETECTED", 
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    self.send_debug_image(debug_frame, msg.header)

def main(args=None):
    rclpy.init(args=args)
    node = ObjectDetectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
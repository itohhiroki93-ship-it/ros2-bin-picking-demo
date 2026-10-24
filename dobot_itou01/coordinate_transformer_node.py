#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from sensor_msgs.msg import Image
import numpy as np
import cv2
from cv_bridge import CvBridge

class CoordinateTransformerNode(Node):
    def __init__(self):
        super().__init__('coordinate_transformer_node')
        
        # 1. ROS 2パラメータの宣言 (初期値の設定)
        default_H = [
            -0.116054, -1.161007, 197.10782,
             0.049821,  0.327266, -58.636398,
            -0.000578, -0.005786,   1.0
        ]
        self.declare_parameter('homography_matrix', default_H)
        self.declare_parameter('target_z', -5.0)

        h_flat = self.get_parameter('homography_matrix').get_parameter_value().double_array_value
        self.target_z = self.get_parameter('target_z').get_parameter_value().double_value

        if len(h_flat) == 9:
            self.H = np.array(h_flat, dtype=np.float64).reshape((3, 3))
        else:
            self.get_logger().error("ホモグラフィ行列の要素数が不正です。デフォルト値を使用します。")
            self.H = np.array(default_H, dtype=np.float64).reshape((3, 3))

        # =========================================================================
        # 🌟 【重要設定】あらかじめ実機で測っておいた ArUcoマーカー の物理座標(mm)
        # 実際の測定値に合わせて、マーカーID と [ロボットX, ロボットY] を書き換えてください！
        # =========================================================================
        self.marker_robot_coords = {
            0: [103.1, -103.2],  # 例: 左上のマーカー (ID: 0)
            1: [201.8,   95.2],  # 例: 右上のマーカー (ID: 1)
            2: [121.9,   89.8],  # 例: 右下のマーカー (ID: 2)
            3: [194.7, -116.6]   # 例: 左下のマーカー (ID: 3)
        }



        # 2. OpenCV ArUco検出器の初期化 (OpenCV 4.x 互換)
        self.bridge = CvBridge()
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.frame_counter = 0  # ログ間引き用タイマー

        # 3. サブスクライバとパブリッシャの作成
        # ① カメラノードから生の映像を受信して ArUco マーカーを検出
        self.image_sub = self.create_subscription(
            Image,
            'camera/image_raw',
            self.image_callback,
            10
        )

        # ② 物体認識ノードから送信されるピクセル座標 (u, v) を受信
        self.subscription = self.create_subscription(
            Point,
            'detected_object_pixel',
            self.pixel_callback,
            10
        )
        
        # ③ 変換後のロボット物理座標 (X, Y, Z) を送信
        self.publisher_ = self.create_publisher(Point, 'target_robot_pose', 10)

        self.get_logger().info("動的座標変換ノードが起動しました。カメラ映像からのホモグラフィ自動更新を開始します！")
        self.get_logger().info(f"初期ホモグラフィ変換行列 H:\n{self.H}")

    def image_callback(self, msg):
        """
        カメラ生映像を受信し、ArUcoマーカーからホモグラフィ行列 H をリアルタイム更新する
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"画像変換エラー: {e}")
            return

        # ArUcoマーカーの検出 (OpenCVバージョン差分吸収)
        try:
            detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
            corners, ids, _ = detector.detectMarkers(cv_image)
        except AttributeError:
            corners, ids, _ = cv2.aruco.detectMarkers(cv_image, self.aruco_dict, parameters=self.aruco_params)

        if ids is None:
            return

        src_pts = []  # カメラのピクセル座標 (u, v)
        dst_pts = []  # ロボットの物理座標 (X, Y)
        detected_ids = ids.flatten()

        # 検出された各マーカーの中心点を計算
        for i, marker_id in enumerate(detected_ids):
            if marker_id in self.marker_robot_coords:
                # 4つの角のピクセル座標から、対角線の交点（中心点）を算出
                corner_points = corners[i][0]
                center_u = np.mean(corner_points[:, 0])
                center_v = np.mean(corner_points[:, 1])

                src_pts.append([center_u, center_v])
                dst_pts.append(self.marker_robot_coords[marker_id])

        # ホモグラフィ行列の計算には最低4点の対応関係が必要！
        if len(src_pts) >= 4:
            src_arr = np.array(src_pts, dtype=np.float32)
            dst_arr = np.array(dst_pts, dtype=np.float32)

            # RANSACアルゴリズムにより、ノイズや誤差を排除して行列を計算
            H_new, _ = cv2.findHomography(src_arr, dst_arr, cv2.RANSAC)

            if H_new is not None:
                self.H = H_new  # 行列 H を最新の状態に上書き！
                
                # 30フレームに1回だけターミナルに更新状態をログ出力
                self.frame_counter += 1
                if self.frame_counter % 30 == 0:
                    self.get_logger().info(
                        f"🔄 [自動補正中] {len(src_pts)}枚のArUcoマーカーから変換行列 H を更新しました。"
                    )

    def pixel_callback(self, msg):
        """
        最新のホモグラフィ行列 H を使って、推論されたピクセル座標を実機ミリメートル座標へ変換
        """
        u = msg.x
        v = msg.y
        
        # 同次座標系 [u, v, 1]
        vec = np.array([u, v, 1.0], dtype=np.float64)
        
        # 最新の行列で計算 H * vec
        res = self.H.dot(vec)
        
        w = res[2]
        if abs(w) < 1e-6:
            self.get_logger().warn("行列計算の結果、分母 w が 0 に近いため、今回の変換をスキップしました。")
            return
        else:
            robot_x = round(res[0] / w, 1)
            robot_y = round(res[1] / w, 1)

        # 変換されたロボット座標をPointメッセージにしてパブリッシュ
        robot_pose_msg = Point()
        robot_pose_msg.x = float(robot_x)
        robot_pose_msg.y = float(robot_y)
        robot_pose_msg.z = float(self.target_z)
        
        self.publisher_.publish(robot_pose_msg)
        self.get_logger().info(
            f"🎯 座標変換完了: Pixel({u:.1f}, {v:.1f}) -> Robot(X:{robot_x:.1f}, Y:{robot_y:.1f}, Z:{self.target_z:.1f})"
        )

def main(args=None):
    rclpy.init(args=args)
    node = CoordinateTransformerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
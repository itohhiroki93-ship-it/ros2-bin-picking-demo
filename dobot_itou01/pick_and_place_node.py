#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from geometry_msgs.msg import Point
from std_msgs.msg import Bool
from dobot_msgs.action import PointToPoint
from dobot_msgs.srv import SuctionCupControl


class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')
        
        # 1. ロボット動作パラメータの宣言 (実験室の環境に合わせて数値を調整できます)
        self.declare_parameter('safe_z', 50.0)      # 移動時の安全上空高さ (mm)
        self.declare_parameter('basket_x', 100.0)   # かごの X 座標 (mm)
        self.declare_parameter('basket_y', -150.0)  # かごの Y 座標 (mm)
        self.declare_parameter('basket_z', -40.0)    # かごの中に落とす高さ (mm)
        self.declare_parameter('home_x', 100.0)     # 待機ホーム位置 X (mm)
        self.declare_parameter('home_y', -150.0)       # 待機ホーム位置 Y (mm)
        self.declare_parameter('home_z', 50.0)      # 待機ホーム位置 Z (mm)

        self.safe_z = self.get_parameter('safe_z').get_parameter_value().double_value
        self.basket_x = self.get_parameter('basket_x').get_parameter_value().double_value
        self.basket_y = self.get_parameter('basket_y').get_parameter_value().double_value
        self.basket_z = self.get_parameter('basket_z').get_parameter_value().double_value
        self.home_x = self.get_parameter('home_x').get_parameter_value().double_value
        self.home_y = self.get_parameter('home_y').get_parameter_value().double_value
        self.home_z = self.get_parameter('home_z').get_parameter_value().double_value

        # 2. パブリッシャの作成 (アームの動作状況: True=動作中, False=停止中)
        self.status_publisher = self.create_publisher(Bool, 'arm_status', 10)
        
        # 3. 移動（PTP）用のアクションクライアント
        self.ptp_action_client = ActionClient(self, PointToPoint, 'PTP_action')
        
        # 吸着ポンプ制御用のサービスクライアント
        self.suction_client = self.create_client(SuctionCupControl, 'dobot_suction_cup_service')
        
        # 4. 座標変換ノードからロボット座標を受信するサブスクライバ
        self.pose_subscription = self.create_subscription(
            Point,
            'target_robot_pose',
            self.pose_callback,
            10
        )
        
        # 状態管理フラグとステップキュー
        self.is_moving = False
        self.sequence_queue = []
        
        self.get_logger().info("🤖 ピック＆プレースノードが起動しました (ホーミング済み前提モード)。")
        
        # ROS 2の通信回線が繋がるのを1秒だけ待ち、即座に「停止中(READY)」をパブリッシュして推論開始！
        self.startup_timer = self.create_timer(1.0, self.startup_timer_callback)

    def startup_timer_callback(self):
        self.startup_timer.cancel()
        self.get_logger().info('✅ スタンバイ完了！AI(YOLO)へ推論許可(READY)の合図を送信します！')
        # 即座に待機状態 (False) に移行し、YOLOに「見てよし！」の合図を送る
        self.publish_arm_status(False)

    def publish_arm_status(self, is_active):
        """
        アームの動作状態をパブリッシュする関数
        is_active: True (動作中 / 移動中), False (停止中 / 待機中)
        """
        self.is_moving = is_active
        msg = Bool()
        msg.data = is_active
        self.status_publisher.publish(msg)
        status_str = "動作中 (BUSY) ➔ 推論停止" if is_active else "停止中 (READY) ➔ 推論再開"
        self.get_logger().info(f"アーム状態を配信: {status_str}")

    def pose_callback(self, msg):
        # 既に移動中の場合は指示を破棄
        if self.is_moving:
            return
        
        target_x = msg.x
        target_y = msg.y
        #target_z = msg.z
        target_z = -63
        
        
        self.get_logger().info(f"🎯 ターゲット座標を受信: X={target_x:.1f}, Y={target_y:.1f}, Z={target_z:.1f}")
        
        # 🌟 即座にアーム動作中 (True) に切り替え！YOLO推論はこれにて一時中断！
        self.publish_arm_status(True)
        
        # =========================================================================
        # 🌟 ピック＆プレース 9段階の自動シーケンスキューを構築！
        # タイプ1: MOVJ(関節補間・高速移動), タイプ2: MOVL(直線補間・垂直昇降)
        # =========================================================================
        self.sequence_queue = [
            # (命令種別, X, Y, Z, 運動タイプ, ログ説明)
            ("MOVE", target_x, target_y, self.safe_z, 1, f"1. ピック位置の上空(Z={self.safe_z})へ移動"),
            ("MOVE", target_x, target_y, target_z,    2, f"2. ピック位置(Z={target_z})へ垂直降下"),
            ("SUCTION", True,                            "3. 【吸着ON】(現在コメントアウト・シミュレーション中)"),
            ("MOVE", target_x, target_y, self.safe_z, 2, f"4. 部品を保持して安全高さ(Z={self.safe_z})まで垂直上昇"),
            ("MOVE", self.basket_x, self.basket_y, self.safe_z, 1, f"5. かごの上空(X={self.basket_x}, Y={self.basket_y})へ高速移動"),
            ("MOVE", self.basket_x, self.basket_y, self.basket_z, 2, f"6. かごの中(Z={self.basket_z})へ垂直降下"),
            ("SUCTION", False,                           "7. 【吸着OFF】(現在コメントアウト・シミュレーション中)"),
            ("MOVE", self.basket_x, self.basket_y, self.safe_z, 2, f"8. かごの上空へ退避上昇"),
            ("MOVE", self.home_x, self.home_y, self.home_z,       1, "9. 待機ホーム位置へ帰還")
        ]
        
        # シーケンスの実行を開始！
        self.execute_next_step()

    def execute_next_step(self):
        """
        キューから次の命令を取り出して実行する状態遷移エンジン
        """
        # キューが空になったら、全工程完了！
        if len(self.sequence_queue) == 0:
            self.get_logger().info("🏁 ピック＆プレースの全シーケンスが完了しました！")
            # アーム停止状態(False)に戻し、YOLOに「次の部品を見てよし！」と伝える
            self.publish_arm_status(False)
            return

        # 次のステップを取り出す
        current_step = self.sequence_queue.pop(0)
        command_type = current_step[0]

        if command_type == "MOVE":
            _, x, y, z, motion_type, desc = current_step
            self.get_logger().info(f"▶️ {desc}")
            self.send_ptp_goal(x, y, z, motion_type)
            
        elif command_type == "SUCTION":
            _, is_on, desc = current_step
            self.get_logger().info(f"▶️ {desc}")
            
            # サービスの存在確認
            if not self.suction_client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error("❌ 吸着サービス 'dobot_suction_cup_service' が利用できません。シーケンスを中断します。")
                self.sequence_queue.clear()
                self.publish_arm_status(False)
                return

            req = SuctionCupControl.Request()
            req.enable_suction = is_on
            self.get_logger().info(f"吸着サービスを呼び出します: enable_suction={is_on}")
            
            # 非同期でサービスコール
            future = self.suction_client.call_async(req)
            future.add_done_callback(self.suction_response_callback)

    def suction_response_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"✅ 吸着サービスコール成功: {response.message}")
        except Exception as e:
            self.get_logger().error(f"❌ 吸着サービスコールの呼び出しに失敗しました: {e}")
        
        # サービスコールの結果が戻ってきたら、次のステップに進む
        self.execute_next_step()

    def send_ptp_goal(self, x, y, z, motion_type=1):
        self.ptp_action_client.wait_for_server()
        
        goal_msg = PointToPoint.Goal()
        goal_msg.target_pose = [float(x), float(y), float(z), 0.0]
        goal_msg.motion_type = motion_type
        goal_msg.velocity_ratio = 0.3      # 安全重視で速度30%
        goal_msg.acceleration_ratio = 0.3  # 加速度30%
        
        send_goal_future = self.ptp_action_client.send_goal_async(goal_msg)
        send_goal_future.add_done_callback(self.ptp_goal_response_callback)

    def ptp_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error('❌ 目標座標への移動が拒否されました。シーケンスを中断します。')
            self.sequence_queue.clear()
            self.publish_arm_status(False)
            return
            
        get_result_future = goal_handle.get_result_async()
        get_result_future.add_done_callback(self.ptp_result_callback)

    def ptp_result_callback(self, future):
        # 1つの移動が完了したら、即座にキューの「次のステップ」を呼び出す！
        self.execute_next_step()

def main(args=None):
    rclpy.init(args=args)
    node = PickAndPlaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
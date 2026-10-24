import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node  # 🌟 これを追加！

def generate_launch_description():
    # ベースとなるフォルダと仮想環境のPythonのパス
    base_dir = os.path.expanduser('~/ros2_ws/src/magician_ros2/dobot_itou01/dobot_itou01')
    venv_python = os.path.join(base_dir, 'venv2', 'bin', 'python3')

    # =========================================================================
    # コマンド・ノードの定義
    # =========================================================================

    # ⑤ Dobot公式システム
    dobot_system = ExecuteProcess(
        cmd=['ros2', 'launch', 'dobot_bringup', 'dobot_magician_control_system.launch.py'],
        additional_env={'MAGICIAN_TOOL': 'suction_cup'},
        output='screen'
    )

    # 🌟 ② rqt_image_view (ROSの正式なNodeとして呼び出す書き方に修正)
    rqt_view = Node(
        package='rqt_image_view',
        executable='rqt_image_view',
        name='rqt_image_view',
        arguments=['/camera/detected_image'], # 🌟 この1行を追加！
        output='screen'
    )

    # ① カメラノード
    camera_node = ExecuteProcess(
        cmd=['python3', os.path.join(base_dir, 'camera_node.py')],
        output='screen'
    )

    # ③ YOLO推論ノード
    yolo_node = ExecuteProcess(
        cmd=[venv_python, os.path.join(base_dir, 'object_detection_node.py')],
        output='screen'
    )

    # ④ 座標変換ノード
    transform_node = ExecuteProcess(
        cmd=[venv_python, os.path.join(base_dir, 'coordinate_transformer_node.py')],
        output='screen'
    )

    # ⑥ ピック＆プレースノード
    pick_place_node = ExecuteProcess(
        cmd=['bash', '-c', f'sleep 8 && python3 {os.path.join(base_dir, "pick_and_place_node.py")}'],
        output='screen'
    )

    # =========================================================================
    # 一斉起動
    # =========================================================================
    return LaunchDescription([
        dobot_system,
        rqt_view,
        camera_node,
        yolo_node,
        transform_node,
        pick_place_node
    ])

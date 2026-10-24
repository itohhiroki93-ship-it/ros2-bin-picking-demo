import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import random
import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import zipfile
from datetime import datetime
import json
import torch
import threading
from ultralytics import SAM, YOLO

# RealSenseカメラ用ライブラリのインポート
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("[情報] pyrealsense2 が検出されませんでした。通常のUSBカメラモードを使用します。")

# ==============================================================================
# SAMセグメンテーション用の設定と状態管理
# ==============================================================================
sam_model = None
current_original_img = None
current_rgba_result = None
camera_window = None
lbl_camera = None
cap = None
camera_pipeline = None  # 🌟 RealSenseカメラ用パイプライン
camera_name_str = ""
preview_window = None
lbl_preview_orig = None
lbl_preview_mask = None
scale_var = None
conf_var = None
is_camera_mode = True

# カメラの画角・ズーム設定 (object_detection_node.py 準拠)
ROI_ZOOM = 2.02
ROI_CROP_LEFT = 0.27
ROI_CROP_RIGHT = 0.01
ROI_CROP_TOP = 0.00
ROI_CROP_BOTTOM = 0.22

# ==============================================================================
# はみ出し対応版：画像合成関数
# ==============================================================================
def place_piece_on_background_2(background_rawdate, piece_rawdate, position,
                              output_path=None, alpha_blend=True, preserve_alpha=True):
    background = background_rawdate
    piece = piece_rawdate

    if background is None or piece is None:
        raise FileNotFoundError("背景画像またはピース画像が読み込めませんでした。")

    bg_ch = background.shape[2] if background.ndim == 3 else 1
    piece_ch = piece.shape[2] if piece.ndim == 3 else 1

    background_f = background.astype(np.float32) / 255.0
    piece_f = piece.astype(np.float32) / 255.0

    if bg_ch == 4:
        bg_rgb = background_f[:, :, :3]
        bg_alpha = background_f[:, :, 3]
    else:
        bg_rgb = background_f[:, :, :3]
        bg_alpha = np.ones(bg_rgb.shape[:2], dtype=np.float32) 

    if piece_ch == 4:
        piece_rgb = piece_f[:, :, :3]
        piece_alpha = piece_f[:, :, 3]
    else:
        piece_rgb = piece_f[:, :, :3]
        piece_alpha = np.ones(piece_rgb.shape[:2], dtype=np.float32) 

    x, y = position
    h, w = piece_rgb.shape[:2]
    H, W = bg_rgb.shape[:2]

    # --- はみ出し部分の自動カット（クロップ）計算 ---
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(W, x + w), min(H, y + h)

    if x1 >= x2 or y1 >= y2:
        if bg_ch == 4 and preserve_alpha:
            return (np.clip(background_f, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            return (np.clip(bg_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    px1, py1 = x1 - x, y1 - y
    px2, py2 = px1 + (x2 - x1), py1 + (y2 - y1)

    bg_roi_rgb = bg_rgb[y1:y2, x1:x2] 
    bg_roi_alpha = bg_alpha[y1:y2, x1:x2]
    piece_roi_rgb = piece_rgb[py1:py2, px1:px2]
    piece_roi_alpha = piece_alpha[py1:py2, px1:px2]

    if alpha_blend:
        A_p = piece_roi_alpha  
        A_b = bg_roi_alpha 
        A_out = A_p + A_b * (1.0 - A_p) 
        pm_piece = piece_roi_rgb * A_p[:, :, None]
        pm_bg = bg_roi_rgb * A_b[:, :, None]
        pm_out = pm_piece + pm_bg * (1.0 - A_p[:, :, None])

        mask = A_out > 1e-6
        out_rgb = np.zeros_like(pm_out)
        out_rgb[mask] = pm_out[mask] / A_out[mask][:, None]
        out_alpha = A_out
    else:
        out_rgb = piece_roi_rgb.copy()
        out_alpha = np.ones((y2-y1, x2-x1), dtype=np.float32)

    if bg_ch == 4 and preserve_alpha:
        combined = background_f.copy()  
        combined[y1:y2, x1:x2, :3] = out_rgb
        combined[y1:y2, x1:x2, 3] = out_alpha
        combined_uint8 = (np.clip(combined, 0.0, 1.0) * 255.0).astype(np.uint8)
    else:
        combined_rgb = bg_rgb.copy()
        combined_rgb[y1:y2, x1:x2, :] = out_rgb
        combined_uint8 = (np.clip(combined_rgb, 0.0, 1.0) * 255.0).astype(np.uint8)

    return combined_uint8

# ==============================================================================
# 既存の画像処理関数群
# ==============================================================================
def resize_piece_image(png_img, new_size_rate):
    alpha = png_img[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return png_img 
    
    x, y, w, h = cv2.boundingRect(coords)
    trimmed = png_img[y:y+h, x:x+w]

    new_size = (max(1, int(trimmed.shape[1] * new_size_rate)), max(1, int(trimmed.shape[0] * new_size_rate))) 
    resized_img = cv2.resize(trimmed, new_size, interpolation=cv2.INTER_AREA)
    return resized_img

def create_mask_from_piece(piece_rawdate):
    if piece_rawdate.shape[2] == 4:
        alpha = piece_rawdate[:, :, 3]
    else:
        gray = cv2.cvtColor(piece_rawdate, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    mask = cv2.threshold(alpha, 1, 255, cv2.THRESH_BINARY)[1]
    return mask

def create_transparent_canvas(bg):
    height, width = bg.shape[:2]
    return np.zeros((height, width, 4), dtype=np.uint8)

def augment_picture(piece_img, hue_shift_range=(-20, 20), sat_scale_range=(0.7, 1.3),
                    alpha_range=(0.8, 1.3), noise_std=5.0, blur_prob=0.5, blur_kernel_size=3):
    if piece_img.shape[2] == 4:
        bgr = piece_img[:, :, :3]
        alpha = piece_img[:, :, 3]
    else:
        bgr = piece_img
        alpha = None
    
    # 1. 色相変化 (Hue Shift)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h_min, h_max = hue_shift_range
    if h_min == h_max:
        hue_shift = h_min
    else:
        hue_shift = random.randint(min(h_min, h_max), max(h_min, h_max))
    hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + hue_shift) % 180
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 2. 彩度変化 (Saturation Scale)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    s_min, s_max = sat_scale_range
    sat_scale = random.uniform(s_min, s_max)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * sat_scale, 0, 255)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 3. 明暗・コントラスト変化 (Brightness / Contrast Alpha)
    a_min, a_max = alpha_range
    alpha_val = random.uniform(a_min, a_max)
    bgr = cv2.convertScaleAbs(bgr, alpha=alpha_val, beta=0)

    # 4. ガウシアンノイズ (Gaussian Noise Standard Deviation)
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, bgr.shape).astype(np.int16)
        bgr = np.clip(bgr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # 5. ぼかし / シャープ化 (Blur / Sharpen Filter)
    if random.random() < blur_prob:
        k = max(1, blur_kernel_size)
        if k % 2 == 0:
            k += 1  # ガウシアンフィルタのカーネルサイズは奇数が必要
        bgr = cv2.GaussianBlur(bgr, (k, k), 0)
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        bgr = cv2.filter2D(bgr, -1, kernel)

    if alpha is not None:
        out = np.dstack([bgr, alpha])
    else:
        out = bgr
    return out

def apply_texture_to_piece(piece_png, textures_folder):
    h, w = piece_png.shape[:2]
    alpha = piece_png[:, :, 3]
    alpha_mask = (alpha > 0).astype(np.uint8)

    texture_files = os.listdir(textures_folder)
    if not texture_files:
        return piece_png
        
    texture_file = random.choice(texture_files)
    texture_path = os.path.join(textures_folder, texture_file)
    texture = cv2.imdecode(np.fromfile(texture_path, dtype=np.uint8), cv2.IMREAD_COLOR)

    if texture is None:
        return piece_png

    H, W = texture.shape[:2]

    if H < h or W < w:
        scale = max(h / H, w / W)
        texture = cv2.resize(texture, (int(W * scale), int(H * scale)))
        H, W = texture.shape[:2]

    y0 = random.randint(0, H - h)
    x0 = random.randint(0, W - w)
    texture_crop = texture[y0:y0 + h, x0:x0 + w]

    result_rgb = texture_crop.copy()
    result_rgb[alpha_mask == 0] = (0, 0, 0)

    result_rgba = np.dstack([result_rgb, alpha])
    return result_rgba

def resize_image(image_array: np.ndarray, target_size: tuple = (512, 512)) -> np.ndarray:
    original_height, original_width = image_array.shape[:2]
    target_width, target_height = target_size

    if target_width < original_width or target_height < original_height:
        interpolation = cv2.INTER_AREA
    else:
        interpolation = cv2.INTER_CUBIC

    resized_img = cv2.resize(image_array, target_size, interpolation=interpolation)
    return resized_img

def rotate_and_crop_piece(image_rgba: np.ndarray, angle_degrees: float) -> np.ndarray:
    original_h, original_w = image_rgba.shape[:2]
    diagonal = int(np.ceil(np.sqrt(original_w**2 + original_h**2)))
    
    canvas_size = diagonal + 4 
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)

    offset_x = (canvas_size - original_w) // 2
    offset_y = (canvas_size - original_h) // 2
    canvas[offset_y:offset_y + original_h, offset_x:offset_x + original_w] = image_rgba

    center = (canvas_size / 2, canvas_size / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle_degrees, 1.0)

    rotated_canvas = cv2.warpAffine(
        canvas, rotation_matrix, (canvas_size, canvas_size),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0)
    )

    alpha = rotated_canvas[:, :, 3]
    coords = cv2.findNonZero(alpha)
    if coords is None:
        return image_rgba 

    x, y, w, h = cv2.boundingRect(coords)
    return rotated_canvas[y:y+h, x:x+w]

def mask_to_yolo_polygons(mask: np.ndarray, class_id: int, min_area: float = 50.0, poly_epsilon_coeff: float = 0.002) -> list:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    yolo_lines = []
    height, width = mask.shape
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        epsilon = poly_epsilon_coeff * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon_str = f"{class_id}"
        for point in approx:
            x, y = point[0]
            norm_x = x / width
            norm_y = y / height
            polygon_str += f" {norm_x:.6f} {norm_y:.6f}"
        yolo_lines.append(polygon_str)
    return yolo_lines

# ==============================================================================
# SAMカメラ撮影と切り抜きプレビュー処理 (SAMsegmentation.py から移植)
# ==============================================================================
def apply_roi(frame):
    h, w = frame.shape[:2]

    try:
        zoom = float(roi_zoom_var.get())
        c_left = float(roi_crop_left_var.get())
        c_right = float(roi_crop_right_var.get())
        c_top = float(roi_crop_top_var.get())
        c_bottom = float(roi_crop_bottom_var.get())
    except Exception:
        zoom = ROI_ZOOM
        c_left = ROI_CROP_LEFT
        c_right = ROI_CROP_RIGHT
        c_top = ROI_CROP_TOP
        c_bottom = ROI_CROP_BOTTOM

    left = int(w * c_left)
    right = int(w * (1.0 - c_right))
    top = int(h * c_top)
    bottom = int(h * (1.0 - c_bottom))
    
    if left >= right or top >= bottom:
        left, right, top, bottom = 0, w, 0, h
        
    cropped = frame[top:bottom, left:right]
    
    if zoom > 1.0:
        ch, cw = cropped.shape[:2]
        new_w, new_h = int(cw / zoom), int(ch / zoom)
        start_x = (cw - new_w) // 2
        start_y = (ch - new_h) // 2
        cropped = cropped[start_y:start_y + new_h, start_x:start_x + new_w]
        
    return cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

def find_external_usb_camera():
    if not os.path.exists("/sys/class/video4linux"):
        for i in range(10):
            cap_test = cv2.VideoCapture(i)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                cap_test.release()
                if ret: return i
        return 0

    integrated_keywords = ["integrated", "internal", "builtin", "laptop", "webcam0", "pc cam", "chicony", "sunplus"]
    for dev in sorted(os.listdir("/sys/class/video4linux")):
        if not dev.startswith("video"): continue
        try:
            idx = int(dev.replace("video", ""))
            name_file = os.path.join("/sys/class/video4linux", dev, "name")
            cam_name = f"video{idx}"
            if os.path.exists(name_file):
                with open(name_file, "r", encoding="utf-8", errors="ignore") as f:
                    cam_name = f.read().strip()
            if any(kw in cam_name.lower() for kw in integrated_keywords):
                continue
            cap_test = cv2.VideoCapture(idx)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                cap_test.release()
                if ret: return idx
        except Exception: continue
    return 0

def open_camera_window():
    global camera_window, lbl_camera, cap, camera_pipeline, camera_name_str

    out_dir = out_entry.get()
    if not out_dir:
        messagebox.showerror("エラー", "先に「出力先Dir」を指定してください。")
        return

    if camera_window is not None and camera_window.winfo_exists():
        camera_window.lift()
        return

    close_camera()

    # 🌟 1. Intel RealSenseカメラを最優先で探索・起動
    is_rs_started = False
    if REALSENSE_AVAILABLE:
        try:
            ctx = rs.context()
            if len(ctx.devices) > 0:
                dev = ctx.devices[0]
                cam_name = dev.get_info(rs.camera_info.name)
                camera_pipeline = rs.pipeline()
                config = rs.config()
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                camera_pipeline.start(config)
                is_rs_started = True
                camera_name_str = f"Intel RealSense ({cam_name})"
                print(f"[情報] RealSenseカメラを初期化しました: {cam_name}")
        except Exception as e:
            print(f"[警告] RealSenseカメラの起動に失敗しました: {e}")
            if camera_pipeline is not None:
                try: camera_pipeline.stop()
                except Exception: pass
                camera_pipeline = None

    # 🌟 2. RealSenseが未検出/起動失敗の場合は、従来のUSBカメラにフォールバック
    if not is_rs_started:
        idx = find_external_usb_camera()
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            messagebox.showerror("エラー", "カメラ (Intel RealSense または 外付けUSBカメラ) に接続できませんでした。")
            close_camera()
            return
        camera_name_str = f"USB/Web Camera (video{idx})"
        print(f"[情報] 外付けUSBカメラ (/dev/video{idx}) を初期化しました。")

    camera_window = tk.Toplevel(root)
    camera_window.title(f"カメラ撮影モニター [{camera_name_str}]")
    camera_window.geometry("700x550")
    camera_window.resizable(False, False)
    camera_window.protocol("WM_DELETE_WINDOW", close_camera)

    lbl_camera = tk.Label(camera_window, bg="black", width=640, height=480)
    lbl_camera.pack(pady=10)

    btn_frame = tk.Frame(camera_window)
    btn_frame.pack()

    tk.Button(btn_frame, text="📸 撮影してSAMで切り抜く", command=capture_and_process, bg="#2196F3", fg="white", font=("", 12, "bold"), width=30, height=2).pack(side=tk.LEFT, padx=10)
    
    update_camera()

def update_camera():
    global cap, camera_pipeline, lbl_camera, camera_window, current_original_img
    if camera_window is None or not camera_window.winfo_exists():
        return
    
    frame = None
    # 🌟 RealSenseからフレームを取得
    if camera_pipeline is not None:
        try:
            frames = camera_pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if color_frame:
                frame = np.asanyarray(color_frame.get_data())
        except Exception as e:
            print(f"[警告] RealSenseフレーム取得エラー: {e}")
    # 🌟 通常USBカメラからフレームを取得
    elif cap is not None and cap.isOpened():
        ret, tmp_frame = cap.read()
        if ret:
            frame = tmp_frame

    if frame is not None:
        processed_frame = apply_roi(frame)
        current_original_img = processed_frame.copy() 
        frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        disp_frame = resize_for_display(frame_rgb, max_size=640)
        img_tk = ImageTk.PhotoImage(Image.fromarray(disp_frame))
        lbl_camera.config(image=img_tk)
        lbl_camera.image = img_tk 
    
    camera_window.after(30, update_camera)

def capture_and_process():
    global is_camera_mode, current_original_img, sam_model
    
    if current_original_img is None:
        messagebox.showerror("エラー", "カメラのフレームが取得できませんでした。")
        return
    
    close_camera()
    is_camera_mode = True

    try:
        if sam_model is None:
            root.title("SAMモデル読込中... お待ちください")
            root.update()
            sam_model = SAM('mobile_sam.pt')
            root.title("YOLO データセット自動生成 ＆ 学習ツール")
        
        open_preview_window()
        run_sam_and_update_preview()
    except Exception as e:
        messagebox.showerror("エラー", f"推論処理に失敗しました。\n\n{str(e)}")

def close_camera():
    global cap, camera_pipeline, camera_window
    if camera_pipeline is not None:
        try:
            camera_pipeline.stop()
        except Exception:
            pass
        camera_pipeline = None
    if cap is not None:
        cap.release()
        cap = None
    if camera_window is not None and camera_window.winfo_exists():
        camera_window.destroy()

def resize_for_display(img_array, max_size=400):
    h, w = img_array.shape[:2]
    scale = min(max_size/w, max_size/h)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

def run_sam_and_update_preview(*args):
    global current_rgba_result
    
    if current_original_img is None: return

    box_scale = scale_var.get()
    conf_thresh = conf_var.get()

    h, w = current_original_img.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    box_w = int(w * box_scale)
    box_h = int(h * box_scale)
    x1 = max(0, center_x - box_w // 2)
    y1 = max(0, center_y - box_h // 2)
    x2 = min(w, center_x + box_w // 2)
    y2 = min(h, center_y + box_h // 2)

    preview_window.title("プレビュー確認 (カメラキャプチャ) - SAM推論中...")
    preview_window.update()

    results = sam_model.predict(
        source=current_original_img,
        bboxes=[[x1, y1, x2, y2]], 
        conf=conf_thresh,
        verbose=False
    )
    result = results[0]

    vis_img = current_original_img.copy()
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 0, 255), max(2, w//200)) 
    vis_img_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)
    
    if result.masks is not None:
        mask_data = result.masks.data[0].cpu().numpy()
        mask = (mask_data * 255).astype(np.uint8)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        current_rgba_result = cv2.cvtColor(current_original_img, cv2.COLOR_BGR2BGRA)
        current_rgba_result[:, :, 3] = mask 
    else:
        current_rgba_result = np.zeros((h, w, 4), dtype=np.uint8)

    disp_vis = resize_for_display(vis_img_rgb)
    disp_rgba = resize_for_display(cv2.cvtColor(current_rgba_result, cv2.COLOR_BGRA2RGBA))

    img_tk_vis = ImageTk.PhotoImage(Image.fromarray(disp_vis))
    img_tk_rgba = ImageTk.PhotoImage(Image.fromarray(disp_rgba))

    lbl_preview_orig.config(image=img_tk_vis)
    lbl_preview_orig.image = img_tk_vis 
    lbl_preview_mask.config(image=img_tk_rgba)
    lbl_preview_mask.image = img_tk_rgba

    preview_window.title("プレビュー確認 (カメラキャプチャ) - 完了")

def save_and_next():
    global current_rgba_result
    if current_rgba_result is None:
        messagebox.showwarning("警告", "保存できる結果がありません。")
        return

    # 保存先フォルダ
    save_dir = os.path.join(BASE_DIR, "picture")
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(save_dir, f"sam_{timestamp}_rgba.png")

    is_success, buffer = cv2.imencode(".png", current_rgba_result)
    if is_success:
        buffer.tofile(out_path)
        
        # ★★★ 新機能: 保存した画像を自動でピースリストへ反映 ★★★
        class_name = f"captured_{timestamp}"
        add_piece_row(default_path=out_path, default_name=class_name)
        
        messagebox.showinfo("保存完了", f"画像を保存し、ピースリストに追加しました！\n{out_path}")
        preview_window.destroy()
        update_preview()
    else:
        messagebox.showerror("エラー", "画像の保存に失敗しました。")

def skip_action():
    preview_window.destroy()

def abort_processing():
    if messagebox.askyesno("確認", "処理を中断してメイン画面に戻りますか？"):
        preview_window.destroy()

def open_preview_window():
    global preview_window, lbl_preview_orig, lbl_preview_mask
    global scale_var, conf_var

    if preview_window is not None and preview_window.winfo_exists():
        preview_window.lift()
        return

    preview_window = tk.Toplevel(root)
    preview_window.geometry("900x650")
    preview_window.resizable(False, False)
    preview_window.protocol("WM_DELETE_WINDOW", abort_processing)

    frame_images = tk.Frame(preview_window)
    frame_images.pack(fill=tk.BOTH, expand=True, pady=10)

    frame_left = tk.Frame(frame_images)
    frame_left.pack(side=tk.LEFT, expand=True)
    tk.Label(frame_left, text="AIに与える枠 (赤枠内を探します)", font=("", 10, "bold")).pack()
    lbl_preview_orig = tk.Label(frame_left, bg="gray", width=400, height=400)
    lbl_preview_orig.pack(padx=10)

    frame_right = tk.Frame(frame_images)
    frame_right.pack(side=tk.LEFT, expand=True)
    tk.Label(frame_right, text="抽出結果 (透過PNG)", font=("", 10, "bold")).pack()
    lbl_preview_mask = tk.Label(frame_right, bg="black", width=400, height=400)
    lbl_preview_mask.pack(padx=10)

    frame_params = tk.LabelFrame(preview_window, text="抽出パラメータの微調整 (うまく切り抜けない場合はここを変更)")
    frame_params.pack(fill=tk.X, padx=20, pady=10)

    scale_var = tk.DoubleVar(value=0.5)
    conf_var = tk.DoubleVar(value=0.4)

    scale_var.trace_add("write", lambda *args: run_sam_and_update_preview())
    conf_var.trace_add("write", lambda *args: run_sam_and_update_preview())

    f_s = tk.Frame(frame_params)
    f_s.pack(fill=tk.X, pady=5)
    tk.Label(f_s, text="枠の大きさ (Box Scale):", width=20, anchor="e").pack(side=tk.LEFT)
    tk.Scale(f_s, variable=scale_var, from_=0.1, to=1.0, resolution=0.05, orient=tk.HORIZONTAL, length=300).pack(side=tk.LEFT, padx=10)

    f_c = tk.Frame(frame_params)
    f_c.pack(fill=tk.X, pady=5)
    tk.Label(f_c, text="確信度閾値 (Confidence):", width=20, anchor="e").pack(side=tk.LEFT)
    tk.Scale(f_c, variable=conf_var, from_=0.1, to=0.9, resolution=0.05, orient=tk.HORIZONTAL, length=300).pack(side=tk.LEFT, padx=10)

    frame_btns = tk.Frame(preview_window)
    frame_btns.pack(fill=tk.X, pady=10)

    tk.Button(frame_btns, text="❌ 中断する", command=abort_processing, width=15, bg="#F44336", fg="white").pack(side=tk.LEFT, padx=30)
    tk.Button(frame_btns, text="▶ キャンセル (保存せず閉じる)", command=skip_action, width=25).pack(side=tk.LEFT, padx=10)
    tk.Button(frame_btns, text="✅ これで保存", command=save_and_next, width=25, height=2, bg="#4CAF50", fg="white", font=("", 10, "bold")).pack(side=tk.RIGHT, padx=30)


# ==============================================================================
# UI用 プレビュー処理 (既存)
# ==============================================================================
preview_window_scale = None
preview_img_min_tk = None
preview_img_max_tk = None

def generate_preview_image(bg_path, piece_path, scale_rate):
    if not bg_path or not piece_path or not os.path.exists(bg_path) or not os.path.exists(piece_path):
        return None
    bg_original = cv2.imdecode(np.fromfile(bg_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    piece_original = cv2.imdecode(np.fromfile(piece_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if bg_original is None or piece_original is None:
        return None

    bg_base = resize_image(bg_original, (512, 512))
    piece_resized = resize_piece_image(piece_original, scale_rate)
    rate_h = 512 / bg_original.shape[0]
    rate_w = 512 / bg_original.shape[1] 
    piece_h = int(piece_resized.shape[0] * rate_h)
    piece_w = int(piece_resized.shape[1] * rate_w)
    piece_final_size = resize_image(piece_resized, (max(1, piece_w), max(1, piece_h)))

    h, w = piece_final_size.shape[:2]
    H, W = bg_base.shape[:2]
    x = max(0, (W - w) // 2)
    y = max(0, (H - h) // 2)
    
    preview_bg = place_piece_on_background_2(bg_base.copy(), piece_final_size, (x, y))
    preview_rgb = cv2.cvtColor(preview_bg, cv2.COLOR_BGR2RGB)
    return Image.fromarray(preview_rgb)

def update_preview(*args):
    global preview_window_scale, preview_img_min_tk, preview_img_max_tk
    if preview_window_scale is None or not preview_window_scale.winfo_exists():
        return
    
    if not bg_entries or not piece_entries:
        return
        
    bg_path = bg_entries[0][1].get().strip()
    piece_path = piece_entries[0][1].get().strip()
    
    min_val = min_scale_var.get()
    max_val = max_scale_var.get()

    img_min = generate_preview_image(bg_path, piece_path, min_val)
    if img_min:
        preview_img_min_tk = ImageTk.PhotoImage(img_min)
        lbl_preview_min.config(image=preview_img_min_tk)
        lbl_preview_min_text.config(text=f"最小スケール ({min_val:.2f}倍)")

    img_max = generate_preview_image(bg_path, piece_path, max_val)
    if img_max:
        preview_img_max_tk = ImageTk.PhotoImage(img_max)
        lbl_preview_max.config(image=preview_img_max_tk)
        lbl_preview_max_text.config(text=f"最大スケール ({max_val:.2f}倍)")

def open_preview():
    global preview_window_scale, lbl_preview_min, lbl_preview_max
    global lbl_preview_min_text, lbl_preview_max_text
    if preview_window_scale is not None and preview_window_scale.winfo_exists():
        preview_window_scale.lift()
        return
    preview_window_scale = tk.Toplevel(root)
    preview_window_scale.title("スケール リアルタイムプレビュー")
    preview_window_scale.geometry("1050x580")
    preview_window_scale.resizable(False, False)

    frame_min = tk.Frame(preview_window_scale)
    frame_min.pack(side=tk.LEFT, padx=10, pady=10)
    lbl_preview_min_text = tk.Label(frame_min, text="最小スケール", font=("sans-serif", 12, "bold"))
    lbl_preview_min_text.pack()
    lbl_preview_min = tk.Label(frame_min, bg="gray", width=512, height=512)
    lbl_preview_min.pack()

    frame_max = tk.Frame(preview_window_scale)
    frame_max.pack(side=tk.LEFT, padx=10, pady=10)
    lbl_preview_max_text = tk.Label(frame_max, text="最大スケール", font=("sans-serif", 12, "bold"))
    lbl_preview_max_text.pack()
    lbl_preview_max = tk.Label(frame_max, bg="gray", width=512, height=512)
    lbl_preview_max.pack()
    update_preview()

# ==============================================================================
# data.yaml 生成処理 (既存)
# ==============================================================================
def generate_data_yaml(output_dir, piece_data_list):
    yaml_path_root = os.path.join(BASE_DIR, "data.yaml")
    yaml_path_dataset = os.path.join(output_dir, "data.yaml")
    
    dataset_dir_f = output_dir.replace("\\", "/")

    yaml_content = f"path: {dataset_dir_f}\n"
    yaml_content += "train: images\n"
    yaml_content += "val: images\n"
    yaml_content += f"nc: {len(piece_data_list)}\n"
    yaml_content += "names:\n"
    
    for i, (_, name) in enumerate(piece_data_list):
        yaml_content += f"  {i}: {name}\n"

    try:
        with open(yaml_path_root, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        os.makedirs(output_dir, exist_ok=True)
        with open(yaml_path_dataset, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        return True
    except Exception as e:
        print(f"data.yaml 生成エラー: {e}")
        return False

def manual_generate_yaml():
    output_dir = out_entry.get()
    piece_data_list = [(p_entry.get().strip(), n_entry.get().strip()) for _, p_entry, n_entry in piece_entries]
    if generate_data_yaml(output_dir, piece_data_list):
        messagebox.showinfo("完了", "data.yaml を生成・更新しました！")
    else:
        messagebox.showerror("エラー", "data.yaml の保存に失敗しました。")

# ==============================================================================
# データセットのクリア・保存処理 (既存)
# ==============================================================================
def clear_dataset():
    output_dir = out_entry.get()
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")

    if not messagebox.askyesno("確認", "現在のデータ（images と labels の中身）を全て削除しますか？\n※この操作は取り消せません。"):
        return

    deleted_files = 0
    for target_dir in [images_dir, labels_dir]:
        if os.path.exists(target_dir):
            for filename in os.listdir(target_dir):
                filepath = os.path.join(target_dir, filename)
                try:
                    if os.path.isfile(filepath):
                        os.remove(filepath)
                        deleted_files += 1
                except Exception as e:
                    print(f"削除エラー {filepath}: {e}")
    
    messagebox.showinfo("完了", f"データをクリアしました。\n（削除されたファイル: {deleted_files}個）")

def save_dataset_to_zip():
    output_dir = out_entry.get()
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    yaml_path = os.path.join(output_dir, "data.yaml")

    bg_paths = [p_entry.get().strip() for _, p_entry in bg_entries if p_entry.get().strip()]
    piece_paths = [p_entry.get().strip() for _, p_entry, _ in piece_entries if p_entry.get().strip()]

    if not os.path.exists(images_dir) or len(os.listdir(images_dir)) == 0:
        messagebox.showwarning("警告", "保存する画像データがありません。\n先にデータセットを生成してください。")
        return

    archives_dir = os.path.join(output_dir, "archives")
    os.makedirs(archives_dir, exist_ok=True)

    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_filename = os.path.join(archives_dir, f"dataset_{now_str}.zip")

    try:
        with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root_dir, _, files in os.walk(images_dir):
                for file in files:
                    file_path = os.path.join(root_dir, file)
                    arcname = os.path.join("images", os.path.relpath(file_path, images_dir))
                    zipf.write(file_path, arcname)
            
            if os.path.exists(labels_dir):
                for root_dir, _, files in os.walk(labels_dir):
                    for file in files:
                        file_path = os.path.join(root_dir, file)
                        arcname = os.path.join("labels", os.path.relpath(file_path, labels_dir))
                        zipf.write(file_path, arcname)

            if os.path.exists(yaml_path):
                zipf.write(yaml_path, "data.yaml")

            for bg_p in bg_paths:
                if os.path.exists(bg_p) and os.path.isfile(bg_p):
                    zipf.write(bg_p, os.path.join("backgrounds", os.path.basename(bg_p)))

            for p_p in piece_paths:
                if os.path.exists(p_p) and os.path.isfile(p_p):
                    zipf.write(p_p, os.path.join("pieces", os.path.basename(p_p)))

            param_info = f"--- Dataset Generation Parameters ({now_str}) ---\n"
            param_info += f"Total Images: {len(os.listdir(images_dir))}\n"
            param_info += f"Mode: {tex_mode_var.get()}\n"
            param_info += f"Pieces per image: {min_pieces_var.get()} to {max_pieces_var.get()}\n"
            param_info += f"Scale range: {min_scale_var.get()} to {max_scale_var.get()}\n"
            param_info += f"Allow Crop (はみ出し許可): {allow_crop_var.get()}\n"
            param_info += f"Rotation Angle Range: {min_angle_var.get()} to {max_angle_var.get()}\n"
            param_info += f"Hue Shift Range: {hue_shift_min_var.get()} to {hue_shift_max_var.get()}\n"
            param_info += f"Saturation Scale Range: {sat_scale_min_var.get()} to {sat_scale_max_var.get()}\n"
            param_info += f"Brightness/Contrast Alpha Range: {alpha_min_var.get()} to {alpha_max_var.get()}\n"
            param_info += f"Noise Std Dev: {noise_std_var.get()}\n"
            param_info += f"Blur Prob / Kernel: {blur_prob_var.get()} / {blur_kernel_var.get()}\n"
            param_info += f"Target Resolution: {target_res_var.get()} px\n"
            param_info += f"Min Contour Area: {min_contour_area_var.get()} px\n"
            param_info += f"Polygon Approx Epsilon: {poly_epsilon_var.get()}\n"
            param_info += f"Camera ROI (Zoom: {roi_zoom_var.get()}, L:{roi_crop_left_var.get()}, R:{roi_crop_right_var.get()}, T:{roi_crop_top_var.get()}, B:{roi_crop_bottom_var.get()})\n"
            param_info += f"\n[Used Assets]\n"
            param_info += f"Backgrounds: {', '.join(bg_paths)}\n"
            for i, (_, p_entry, n_entry) in enumerate(piece_entries):
                param_info += f"Piece {i+1} (Class {i} '{n_entry.get()}'): {p_entry.get()}\n"
            
            zipf.writestr("parameters.txt", param_info)

        messagebox.showinfo("保存完了", f"データセットと使用素材をZIPとして保存しました。\n\n保存先:\n{zip_filename}")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("エラー", f"保存中にエラーが発生しました。\n\n{str(e)}")


# ==============================================================================
# コア処理：データ合成生成本体の処理関数 (process_dataset からの切り出し)
# ==============================================================================
def generate_dataset_files():
    bg_paths = [p_entry.get().strip() for _, p_entry in bg_entries if p_entry.get().strip()]
    piece_data_list = [(p_entry.get().strip(), n_entry.get().strip()) for _, p_entry, n_entry in piece_entries]
    textures_dir = tex_entry.get()
    output_dir = out_entry.get()
    mode_selection = tex_mode_var.get()
    is_crop_allowed = allow_crop_var.get()
    
    num_dataset_pairs = int(num_entry.get())
    min_p = int(min_pieces_var.get())
    max_p = int(max_pieces_var.get())
    min_s = float(min_scale_var.get())
    max_s = float(max_scale_var.get())

    # UIから10個の新パラメータを取得
    min_angle = float(min_angle_var.get())
    max_angle = float(max_angle_var.get())
    hue_shift_min = int(hue_shift_min_var.get())
    hue_shift_max = int(hue_shift_max_var.get())
    sat_scale_min = float(sat_scale_min_var.get())
    sat_scale_max = float(sat_scale_max_var.get())
    alpha_min = float(alpha_min_var.get())
    alpha_max = float(alpha_max_var.get())
    noise_std = float(noise_std_var.get())
    blur_prob = float(blur_prob_var.get())
    blur_kernel_size = int(blur_kernel_var.get())
    target_res = int(target_res_var.get())
    min_contour_area = float(min_contour_area_var.get())
    poly_epsilon = float(poly_epsilon_var.get())

    output_images_dir = os.path.join(output_dir, "images")
    output_labels_dir = os.path.join(output_dir, "labels")
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)

    generate_data_yaml(output_dir, piece_data_list)

    piece_sources = []
    for class_id, (p_path, _) in enumerate(piece_data_list):
        img = cv2.imdecode(np.fromfile(p_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"ピース画像の読み込みに失敗しました: {p_path}")
        piece_sources.append({"img": img, "class_id": class_id})

    existing_files = len([f for f in os.listdir(output_images_dir) if f.endswith('.jpg')])

    for img_idx in range(num_dataset_pairs):
        # メインスレッドでのGUI文字更新
        exec_btn.config(text=f"生成中... ({img_idx+1}/{num_dataset_pairs})")
        root.update()

        selected_bg_path = random.choice(bg_paths)
        bg_original = cv2.imdecode(np.fromfile(selected_bg_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        bg_base = resize_image(bg_original, (target_res, target_res))

        current_bg = augment_picture(
            bg_base,
            hue_shift_range=(hue_shift_min, hue_shift_max),
            sat_scale_range=(sat_scale_min, sat_scale_max),
            alpha_range=(alpha_min, alpha_max),
            noise_std=noise_std,
            blur_prob=blur_prob,
            blur_kernel_size=blur_kernel_size
        )
        num_pieces = random.randint(min_p, max_p)
        placed_pieces_info = []

        for p_idx in range(num_pieces):
            chosen_piece = random.choice(piece_sources)
            piece_original = chosen_piece["img"]
            class_id = chosen_piece["class_id"]

            random_rate = random.uniform(min_s, max_s)
            piece_resized = resize_piece_image(piece_original, random_rate)
            
            rate_h = target_res / bg_original.shape[0]
            rate_w = target_res / bg_original.shape[1] 
            piece_h = int(piece_resized.shape[0] * rate_h)
            piece_w = int(piece_resized.shape[1] * rate_w)
            
            random_angle = random.uniform(min_angle, max_angle)
            piece_rotated = rotate_and_crop_piece(piece_resized, random_angle)
            piece_final_size = resize_image(piece_rotated, (max(1, piece_w), max(1, piece_h)))

            if mode_selection == "texture":
                piece_textured = apply_texture_to_piece(piece_final_size, textures_dir)
            else:
                piece_textured = piece_final_size
                
            piece_ready = augment_picture(
                piece_textured,
                hue_shift_range=(hue_shift_min, hue_shift_max),
                sat_scale_range=(sat_scale_min, sat_scale_max),
                alpha_range=(alpha_min, alpha_max),
                noise_std=noise_std,
                blur_prob=blur_prob,
                blur_kernel_size=blur_kernel_size
            )

            h, w = piece_ready.shape[:2]
            H, W = current_bg.shape[:2]

            if is_crop_allowed:
                u = random.randint(-w // 2, W - w // 2)
                v = random.randint(-h // 2, H - h // 2)
            else:
                max_u = max(0, W - w)
                max_v = max(0, H - h)
                u = random.randint(0, max_u)
                v = random.randint(0, max_v)

            position = (u, v)

            current_bg = place_piece_on_background_2(current_bg, piece_ready, position)

            transparent_canvas = create_transparent_canvas(current_bg)
            transparent_combined = place_piece_on_background_2(transparent_canvas, piece_ready, position)
            current_mask = create_mask_from_piece(transparent_combined)

            mask_inv = cv2.bitwise_not(current_mask) 
            for past_info in placed_pieces_info:
                past_info["mask"] = cv2.bitwise_and(past_info["mask"], mask_inv)

            placed_pieces_info.append({
                "class_id": class_id,
                "mask": current_mask
            })

        file_number = existing_files + img_idx
        filename_base = f"image_{file_number:04d}"
        
        yolo_label_lines = []
        for info in placed_pieces_info:
            polygons = mask_to_yolo_polygons(info["mask"], info["class_id"], min_area=min_contour_area, poly_epsilon_coeff=poly_epsilon)
            yolo_label_lines.extend(polygons)

        img_save_path = os.path.join(output_images_dir, f"{filename_base}.jpg")
        cv2.imencode(".jpg", current_bg)[1].tofile(img_save_path)
        
        txt_save_path = os.path.join(output_labels_dir, f"{filename_base}.txt")
        with open(txt_save_path, "w", encoding="utf-8") as f:
            for line in yolo_label_lines:
                f.write(f"{line}\n")
    return True

# ==============================================================================
# 新機能：データ生成 ＆ YOLO学習 一括実行パイプラインの管理
# ==============================================================================
def start_pipeline():
    # 入力バリデーション
    bg_paths = [p_entry.get().strip() for _, p_entry in bg_entries if p_entry.get().strip()]
    if not bg_paths:
        messagebox.showerror("エラー", "背景画像が1つも設定されていません。")
        return
    for path in bg_paths:
        if not os.path.exists(path):
            messagebox.showerror("エラー", f"背景画像が見つかりません:\n{path}")
            return
            
    piece_data_list = [(p_entry.get().strip(), n_entry.get().strip()) for _, p_entry, n_entry in piece_entries]
    if not piece_data_list:
        messagebox.showerror("エラー", "ピース画像が1つも設定されていません。")
        return
    for i, (path, name) in enumerate(piece_data_list):
        if not path or not os.path.exists(path):
            messagebox.showerror("エラー", f"ピース画像 {i+1} ({name}) が見つかりません:\n{path}")
            return

    try:
        _ = int(num_entry.get())
        _ = int(min_pieces_var.get())
        _ = int(max_pieces_var.get())
        _ = float(min_scale_var.get())
        _ = float(max_scale_var.get())
        _ = int(epochs_var.get())
        _ = int(batch_var.get())
        _ = float(min_angle_var.get())
        _ = float(max_angle_var.get())
        _ = int(hue_shift_min_var.get())
        _ = int(hue_shift_max_var.get())
        _ = float(sat_scale_min_var.get())
        _ = float(sat_scale_max_var.get())
        _ = float(alpha_min_var.get())
        _ = float(alpha_max_var.get())
        _ = float(noise_std_var.get())
        _ = float(blur_prob_var.get())
        _ = int(blur_kernel_var.get())
        _ = int(target_res_var.get())
        _ = float(min_contour_area_var.get())
        _ = float(poly_epsilon_var.get())
        _ = float(roi_zoom_var.get())
        _ = float(roi_crop_left_var.get())
        _ = float(roi_crop_right_var.get())
        _ = float(roi_crop_top_var.get())
        _ = float(roi_crop_bottom_var.get())
    except ValueError:
        messagebox.showerror("エラー", "設定入力に不正な文字が含まれています。半角数字で入力してください。")
        return

    # ボタンを一時的に無効化
    exec_btn.config(text="データ生成中... (裏の画面を確認)", state=tk.DISABLED)
    root.update()

    # GUIがフリーズするのを防ぐため、別スレッドで処理を走らせる
    threading.Thread(target=run_pipeline_thread, daemon=True).start()

def run_pipeline_thread():
    try:
        output_dir = out_entry.get()
        num_dataset_pairs = int(num_entry.get())
        
        # 1. データセット生成
        print("\n--- [STEP 1/2] データセット合成生成を開始します ---")
        generate_dataset_files()
        print("\n✅ データセットの生成が完了しました！")
        
        # 2. YOLOv8 学習
        print("\n--- [STEP 2/2] YOLOv8 セグメンテーションモデルの学習を開始します ---")
        exec_btn.config(text="AI 学習実行中... (裏の黒い画面を確認)")
        root.update()
        
        epochs_val = int(epochs_var.get())
        batch_val = int(batch_var.get())
        target_res_val = int(target_res_var.get())
        yaml_path = os.path.join(output_dir, "data.yaml")
        
        if torch.cuda.is_available():
            device = "0"
            device_name = torch.cuda.get_device_name(0)
            print(f"\n✅ GPUが検出されました: {device_name} を使用して学習します。")
        else:
            device = "cpu"
            print("\n⚠️ GPUが検出されませんでした。CPUで学習を行います。")
            
        model = YOLO("yolov8n-seg.pt") 
        print(f"学習パラメータ: data={yaml_path}, epochs={epochs_val}, batch={batch_val}, imgsz={target_res_val}, device={device}")
        
        # Windows/Linuxのマルチプロセッシングエラー回避のため workers=0 とする
        model.train(
            data=yaml_path,
            epochs=epochs_val,
            imgsz=target_res_val,
            batch=batch_val,
            device=device,
            workers=0 
        )
        
        print("\n🎉 すべてのパイプライン処理が正常に完了しました！")
        messagebox.showinfo(
            "完了", 
            f"{num_dataset_pairs} ペアのデータセット生成とYOLOv8の学習が完了しました！\n\n"
            f"学習済みモデル (best.pt) は runs/detect/train/weights に保存されました。"
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        messagebox.showerror("エラー", f"処理中にエラーが発生しました。\n\n{str(e)}")
    finally:
        exec_btn.config(text="🚀 データ生成 ＆ YOLO学習 実行", state=tk.NORMAL)

# 後方互換性のためのダミー
def process_dataset():
    start_pipeline()

# ==============================================================================
# ★ 設定（パス）の保存・読込処理 (既存)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}

def save_default_path(key, entry_widget):
    current_path = entry_widget.get()
    config = load_config()
    config[key] = current_path
    
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("保存完了", "デフォルトパスを保存しました。\n次回起動時からこのパスが初期表示されます。")
    except Exception as e:
        messagebox.showerror("エラー", f"設定の保存に失敗しました。\n{str(e)}")

def save_all_bg_defaults():
    bg_paths = [path_entry.get().strip() for _, path_entry in bg_entries if path_entry.get().strip()]
    config = load_config()
    config["bg_paths"] = bg_paths
    
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("保存完了", "現在の背景一覧の構成をデフォルトとして保存しました。")
    except Exception as e:
        messagebox.showerror("エラー", f"設定の保存に失敗しました。\n{str(e)}")

def save_all_piece_defaults():
    piece_data = [{"path": p_entry.get().strip(), "name": n_entry.get().strip()} for _, p_entry, n_entry in piece_entries]
    config = load_config()
    config["piece_data"] = piece_data
    
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("保存完了", "現在のピース一覧の構成（名前とパス）をデフォルトとして保存しました。")
    except Exception as e:
        messagebox.showerror("エラー", f"設定の保存に失敗しました。\n{str(e)}")

# ==============================================================================
# UIコンポーネント用ヘルパー (既存)
# ==============================================================================
def select_file(entry_widget, title, filetypes):
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if file_path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)
        update_preview()

def select_dir(entry_widget, title):
    dir_path = filedialog.askdirectory(title=title)
    if dir_path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, dir_path)

def toggle_tex_state():
    if tex_mode_var.get() == "original":
        tex_entry.configure(state='disabled')
        tex_btn.configure(state='disabled')
        tex_default_btn.configure(state='disabled')
    else:
        tex_entry.configure(state='normal')
        tex_btn.configure(state='normal')
        tex_default_btn.configure(state='normal')

def save_all_domain_defaults():
    config = load_config()
    try:
        config["min_angle"] = float(min_angle_var.get())
        config["max_angle"] = float(max_angle_var.get())
        config["hue_shift_min"] = int(hue_shift_min_var.get())
        config["hue_shift_max"] = int(hue_shift_max_var.get())
        config["sat_scale_min"] = float(sat_scale_min_var.get())
        config["sat_scale_max"] = float(sat_scale_max_var.get())
        config["alpha_min"] = float(alpha_min_var.get())
        config["alpha_max"] = float(alpha_max_var.get())
        config["noise_std"] = float(noise_std_var.get())
        config["blur_prob"] = float(blur_prob_var.get())
        config["blur_kernel"] = int(blur_kernel_var.get())
        config["target_res"] = int(target_res_var.get())
        config["min_contour_area"] = float(min_contour_area_var.get())
        config["poly_epsilon"] = float(poly_epsilon_var.get())
        config["roi_zoom"] = float(roi_zoom_var.get())
        config["roi_crop_left"] = float(roi_crop_left_var.get())
        config["roi_crop_right"] = float(roi_crop_right_var.get())
        config["roi_crop_top"] = float(roi_crop_top_var.get())
        config["roi_crop_bottom"] = float(roi_crop_bottom_var.get())

        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=4)
        messagebox.showinfo("保存完了", "ランダム化・アノテーション・ROI等の詳細設定をデフォルトとして保存しました。")
    except Exception as e:
        messagebox.showerror("エラー", f"設定の保存に失敗しました。\n{str(e)}")

# ==============================================================================
# UIコンポーネント用ヘルパー (既存)
# ==============================================================================
def select_file(entry_widget, title, filetypes):
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    if file_path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, file_path)
        update_preview()

def select_dir(entry_widget, title):
    dir_path = filedialog.askdirectory(title=title)
    if dir_path:
        entry_widget.delete(0, tk.END)
        entry_widget.insert(0, dir_path)

def toggle_tex_state():
    if tex_mode_var.get() == "original":
        tex_entry.configure(state='disabled')
        tex_btn.configure(state='disabled')
        tex_default_btn.configure(state='disabled')
    else:
        tex_entry.configure(state='normal')
        tex_btn.configure(state='normal')
        tex_default_btn.configure(state='normal')

# ==============================================================================
# GUI画面構築
# ==============================================================================
root = tk.Tk()
root.title("YOLO データセット自動生成 ＆ 学習ツール")
root.geometry("880x950") 
root.resizable(True, True)

# メインスクロールバー付きコンテナの設定
main_canvas = tk.Canvas(root, borderwidth=0)
scrollbar = tk.Scrollbar(root, orient="vertical", command=main_canvas.yview)
scrollable_frame = tk.Frame(main_canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
)

canvas_window = main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")

def _on_canvas_configure(event):
    main_canvas.itemconfig(canvas_window, width=event.width)

main_canvas.bind("<Configure>", _on_canvas_configure)
main_canvas.configure(yscrollcommand=scrollbar.set)

# マウスホイールでのスクロール対応 (Linux/Windows)
def _on_mousewheel(event):
    if event.num == 4:
        main_canvas.yview_scroll(-1, "units")
    elif event.num == 5:
        main_canvas.yview_scroll(1, "units")
    else:
        main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

root.bind_all("<MouseWheel>", _on_mousewheel)
root.bind_all("<Button-4>", _on_mousewheel)
root.bind_all("<Button-5>", _on_mousewheel)

main_canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

config_data = load_config()

default_bg_data = config_data.get("bg_paths", [])
if not default_bg_data:
    old_bg = config_data.get("bg_path", os.path.join(BASE_DIR, "picture", "buckGroubnd.jpg"))
    default_bg_data = [old_bg]

default_tex = config_data.get("tex_path", os.path.join(BASE_DIR, "puzzle"))
default_out = config_data.get("out_path", os.path.join(BASE_DIR, "dataset"))

default_piece_data = config_data.get("piece_data", [])
if not default_piece_data:
    old_paths = config_data.get("piece_paths", [])
    if old_paths:
        default_piece_data = [{"path": p, "name": f"class_{i}"} for i, p in enumerate(old_paths)]
    else:
        default_piece_data = [
            {"path": os.path.join(BASE_DIR, "picture", "task5_rgba.png"), "name": "1yen"},
            {"path": os.path.join(BASE_DIR, "picture", "task6_rgba.png"), "name": "10yen"}
        ]

# --- 10個の新パラメータ用変数の初期化 ---
min_angle_var = tk.StringVar(value=str(config_data.get("min_angle", 0.0)))
max_angle_var = tk.StringVar(value=str(config_data.get("max_angle", 360.0)))

hue_shift_min_var = tk.StringVar(value=str(config_data.get("hue_shift_min", -20)))
hue_shift_max_var = tk.StringVar(value=str(config_data.get("hue_shift_max", 20)))

sat_scale_min_var = tk.StringVar(value=str(config_data.get("sat_scale_min", 0.7)))
sat_scale_max_var = tk.StringVar(value=str(config_data.get("sat_scale_max", 1.3)))

alpha_min_var = tk.StringVar(value=str(config_data.get("alpha_min", 0.8)))
alpha_max_var = tk.StringVar(value=str(config_data.get("alpha_max", 1.3)))

noise_std_var = tk.StringVar(value=str(config_data.get("noise_std", 5.0)))

blur_prob_var = tk.StringVar(value=str(config_data.get("blur_prob", 0.5)))
blur_kernel_var = tk.StringVar(value=str(config_data.get("blur_kernel", 3)))

target_res_var = tk.StringVar(value=str(config_data.get("target_res", 512)))

min_contour_area_var = tk.StringVar(value=str(config_data.get("min_contour_area", 50.0)))

poly_epsilon_var = tk.StringVar(value=str(config_data.get("poly_epsilon", 0.002)))

roi_zoom_var = tk.StringVar(value=str(config_data.get("roi_zoom", 2.02)))
roi_crop_left_var = tk.StringVar(value=str(config_data.get("roi_crop_left", 0.27)))
roi_crop_right_var = tk.StringVar(value=str(config_data.get("roi_crop_right", 0.01)))
roi_crop_top_var = tk.StringVar(value=str(config_data.get("roi_crop_top", 0.00)))
roi_crop_bottom_var = tk.StringVar(value=str(config_data.get("roi_crop_bottom", 0.22)))

# --- ファイルパス設定セクション ---
frame_paths = tk.LabelFrame(scrollable_frame, text="ファイルとフォルダの設定")
frame_paths.pack(fill=tk.X, padx=10, pady=5)

def make_path_row(parent, label_text, default_val, btn_cmd, config_key=None):
    frame = tk.Frame(parent)
    frame.pack(fill=tk.X, padx=5, pady=5)
    tk.Label(frame, text=label_text, width=15, anchor="w").pack(side=tk.LEFT)
    entry = tk.Entry(frame)
    entry.insert(0, default_val)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    btn = tk.Button(frame, text="参照...", command=btn_cmd)
    btn.pack(side=tk.LEFT, padx=(0, 5))
    default_btn = None
    if config_key:
        default_btn = tk.Button(frame, text="デフォルトにする", command=lambda: save_default_path(config_key, entry))
        default_btn.pack(side=tk.LEFT)
    return entry, btn, default_btn

# 背景画像の動的リスト化
frame_bgs = tk.Frame(frame_paths)
frame_bgs.pack(fill=tk.X, padx=0, pady=0)
bg_entries = []

def add_bg_row(default_path=""):
    idx = len(bg_entries) + 1
    row_frame = tk.Frame(frame_bgs)
    row_frame.pack(fill=tk.X, padx=5, pady=2)
    tk.Label(row_frame, text=f"背景画像 {idx}", width=14, anchor="w").pack(side=tk.LEFT)
    path_entry = tk.Entry(row_frame)
    path_entry.insert(0, default_path)
    path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    tk.Button(row_frame, text="参照...", command=lambda e=path_entry: select_file(e, f"背景{idx}を選択", [("画像", "*.jpg *.png *.jpeg"), ("All Files", "*.*")])).pack(side=tk.LEFT, padx=(0, 5))
    bg_entries.append((row_frame, path_entry))

def remove_bg_row():
    if len(bg_entries) > 1:
        row_frame, _ = bg_entries.pop()
        row_frame.destroy()
        update_preview()

for bg_p in default_bg_data:
    add_bg_row(bg_p)

frame_bg_ctrl = tk.Frame(frame_paths)
frame_bg_ctrl.pack(fill=tk.X, padx=5, pady=5)
tk.Button(frame_bg_ctrl, text="＋ 背景を追加", command=lambda: add_bg_row(""), width=15).pack(side=tk.LEFT, padx=5)
tk.Button(frame_bg_ctrl, text="－ 最後の背景を削除", command=remove_bg_row, width=18).pack(side=tk.LEFT, padx=5)
tk.Button(frame_bg_ctrl, text="この構成をデフォルトにする", command=save_all_bg_defaults).pack(side=tk.LEFT, padx=15)

# ピース画像の動的リスト化
frame_pieces = tk.Frame(frame_paths)
frame_pieces.pack(fill=tk.X, padx=0, pady=0)
piece_entries = []

def add_piece_row(default_path="", default_name=""):
    idx = len(piece_entries) + 1
    row_frame = tk.Frame(frame_pieces)
    row_frame.pack(fill=tk.X, padx=5, pady=2)
    tk.Label(row_frame, text=f"ピース{idx} (ID:{idx-1})", width=14, anchor="w").pack(side=tk.LEFT)
    tk.Label(row_frame, text="名前:").pack(side=tk.LEFT)
    name_entry = tk.Entry(row_frame, width=10)
    name_entry.insert(0, default_name if default_name else f"class_{idx-1}")
    name_entry.pack(side=tk.LEFT, padx=(0, 5))
    path_entry = tk.Entry(row_frame)
    path_entry.insert(0, default_path)
    path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    tk.Button(row_frame, text="参照...", command=lambda e=path_entry: select_file(e, f"ピース{idx}を選択", [("PNG", "*.png"), ("All Files", "*.*")])).pack(side=tk.LEFT, padx=(0, 5))
    piece_entries.append((row_frame, path_entry, name_entry))

def remove_piece_row():
    if len(piece_entries) > 1:
        row_frame, _, _ = piece_entries.pop()
        row_frame.destroy()
        update_preview()

for p_data in default_piece_data:
    add_piece_row(default_path=p_data.get("path", ""), default_name=p_data.get("name", ""))

frame_piece_ctrl = tk.Frame(frame_paths)
frame_piece_ctrl.pack(fill=tk.X, padx=5, pady=5)
tk.Button(frame_piece_ctrl, text="＋ ピースを追加", command=lambda: add_piece_row("", ""), width=15).pack(side=tk.LEFT, padx=5)
tk.Button(frame_piece_ctrl, text="－ 最後のピースを削除", command=remove_piece_row, width=18).pack(side=tk.LEFT, padx=5)
tk.Button(frame_piece_ctrl, text="📷 カメラ撮影＆SAMで切り抜く", command=open_camera_window, width=28, bg="#FF9800", fg="white", font=("sans-serif", 9, "bold")).pack(side=tk.LEFT, padx=5)
tk.Button(frame_piece_ctrl, text="この構成をデフォルトにする", command=save_all_piece_defaults).pack(side=tk.LEFT, padx=15)

# その他の設定
frame_mode = tk.Frame(frame_paths)
frame_mode.pack(fill=tk.X, padx=5, pady=2)
tex_mode_var = tk.StringVar(value="texture")
tk.Radiobutton(frame_mode, text="モード1: テクスチャ画像でランダム柄にする", variable=tex_mode_var, value="texture", command=toggle_tex_state).pack(anchor="w")
tk.Radiobutton(frame_mode, text="モード2: 元のデザインをそのまま使用する", variable=tex_mode_var, value="original", command=toggle_tex_state).pack(anchor="w")

tex_entry, tex_btn, tex_default_btn = make_path_row(frame_paths, "テクスチャDir:", default_tex, lambda: select_dir(tex_entry, "テクスチャ"), "tex_path")
toggle_tex_state()
out_entry, _, _ = make_path_row(frame_paths, "出力先Dir:", default_out, lambda: select_dir(out_entry, "出力先"), "out_path")

# --- 基本データ生成パラメータ設定セクション ---
frame_params = tk.LabelFrame(scrollable_frame, text="基本データ生成パラメータ")
frame_params.pack(fill=tk.X, padx=10, pady=5)

frame_count = tk.Frame(frame_params)
frame_count.pack(fill=tk.X, padx=5, pady=5)
tk.Label(frame_count, text="1枚あたりの配置個数:", width=22, anchor="w").pack(side=tk.LEFT)
min_pieces_var = tk.StringVar(value="3")
max_pieces_var = tk.StringVar(value="8")
tk.Entry(frame_count, textvariable=min_pieces_var, width=5).pack(side=tk.LEFT)
tk.Label(frame_count, text="～").pack(side=tk.LEFT, padx=5)
tk.Entry(frame_count, textvariable=max_pieces_var, width=5).pack(side=tk.LEFT)
tk.Label(frame_count, text="個").pack(side=tk.LEFT, padx=5)

frame_scale = tk.Frame(frame_params)
frame_scale.pack(fill=tk.X, padx=5, pady=5)
min_scale_var = tk.DoubleVar(value=0.3)
max_scale_var = tk.DoubleVar(value=0.6)
min_scale_var.trace_add("write", update_preview)
max_scale_var.trace_add("write", update_preview)

f_s_min = tk.Frame(frame_scale)
f_s_min.pack(fill=tk.X, pady=2)
tk.Label(f_s_min, text="ピースの最小スケール:", width=22, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_s_min, textvariable=min_scale_var, width=5).pack(side=tk.LEFT)
tk.Scale(f_s_min, variable=min_scale_var, from_=0.1, to=10.0, resolution=0.01, orient=tk.HORIZONTAL, length=300, showvalue=0).pack(side=tk.LEFT, padx=10)

f_s_max = tk.Frame(frame_scale)
f_s_max.pack(fill=tk.X, pady=2)
tk.Label(f_s_max, text="ピースの最大スケール:", width=22, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_s_max, textvariable=max_scale_var, width=5).pack(side=tk.LEFT)
tk.Scale(f_s_max, variable=max_scale_var, from_=0.1, to=10.0, resolution=0.01, orient=tk.HORIZONTAL, length=300, showvalue=0).pack(side=tk.LEFT, padx=10)

btn_preview = tk.Button(frame_scale, text="🔍 大きさプレビューを表示", command=open_preview, bg="#e0f7fa")
btn_preview.pack(pady=5)

frame_crop = tk.Frame(frame_params)
frame_crop.pack(fill=tk.X, padx=5, pady=5)
allow_crop_var = tk.BooleanVar(value=True) 
tk.Checkbutton(frame_crop, text="☑ ピースが画面からはみ出す配置を許可する (はみ出た部分は自動カット)", variable=allow_crop_var, font=("sans-serif", 9, "bold")).pack(side=tk.LEFT)

frame_num = tk.Frame(frame_params)
frame_num.pack(fill=tk.X, padx=5, pady=5)
tk.Label(frame_num, text="合計の生成ペア画像数:", width=22, anchor="w").pack(side=tk.LEFT)
num_entry = tk.Entry(frame_num, width=8)
num_entry.insert(0, "100")
num_entry.pack(side=tk.LEFT)
tk.Label(frame_num, text="枚").pack(side=tk.LEFT, padx=5)

# --- 🎨 ドメインランダム化 パラメータ設定セクション (新規追加) ---
frame_domain = tk.LabelFrame(scrollable_frame, text="🎨 ドメインランダム化 パラメータ設定", fg="#2e7d32", font=("sans-serif", 10, "bold"))
frame_domain.pack(fill=tk.X, padx=10, pady=5)

# 1. ピースの回転角度範囲
f_angle = tk.Frame(frame_domain)
f_angle.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_angle, text="1. ピース回転角度範囲:", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_angle, textvariable=min_angle_var, width=6).pack(side=tk.LEFT)
tk.Label(f_angle, text="～").pack(side=tk.LEFT, padx=3)
tk.Entry(f_angle, textvariable=max_angle_var, width=6).pack(side=tk.LEFT)
tk.Label(f_angle, text="度 (0~360等)").pack(side=tk.LEFT, padx=5)

# 2. 色相変化の幅
f_hue = tk.Frame(frame_domain)
f_hue.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_hue, text="2. 色相変化の幅 (Hue Shift):", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_hue, textvariable=hue_shift_min_var, width=6).pack(side=tk.LEFT)
tk.Label(f_hue, text="～").pack(side=tk.LEFT, padx=3)
tk.Entry(f_hue, textvariable=hue_shift_max_var, width=6).pack(side=tk.LEFT)
tk.Label(f_hue, text="(デフォルト: -20 ～ +20)").pack(side=tk.LEFT, padx=5)

# 3. 彩度変化の倍率
f_sat = tk.Frame(frame_domain)
f_sat.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_sat, text="3. 彩度変化倍率 (Sat Scale):", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_sat, textvariable=sat_scale_min_var, width=6).pack(side=tk.LEFT)
tk.Label(f_sat, text="～").pack(side=tk.LEFT, padx=3)
tk.Entry(f_sat, textvariable=sat_scale_max_var, width=6).pack(side=tk.LEFT)
tk.Label(f_sat, text="倍 (デフォルト: 0.7 ～ 1.3)").pack(side=tk.LEFT, padx=5)

# 4. 明暗・コントラストの変化倍率
f_alpha = tk.Frame(frame_domain)
f_alpha.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_alpha, text="4. 明暗・コントラスト倍率:", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_alpha, textvariable=alpha_min_var, width=6).pack(side=tk.LEFT)
tk.Label(f_alpha, text="～").pack(side=tk.LEFT, padx=3)
tk.Entry(f_alpha, textvariable=alpha_max_var, width=6).pack(side=tk.LEFT)
tk.Label(f_alpha, text="倍 (デフォルト: 0.8 ～ 1.3)").pack(side=tk.LEFT, padx=5)

# 5. ガウシアンノイズの強度
f_noise = tk.Frame(frame_domain)
f_noise.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_noise, text="5. ノイズ強度 (標準偏差 SD):", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_noise, textvariable=noise_std_var, width=6).pack(side=tk.LEFT)
tk.Label(f_noise, text="(デフォルト: 5.0, 0で無効化)").pack(side=tk.LEFT, padx=5)

# 6. ぼかし/シャープ化フィルタ
f_blur = tk.Frame(frame_domain)
f_blur.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_blur, text="6. ぼかし確率 / カーネル:", width=24, anchor="w").pack(side=tk.LEFT)
tk.Label(f_blur, text="確率:").pack(side=tk.LEFT)
tk.Entry(f_blur, textvariable=blur_prob_var, width=6).pack(side=tk.LEFT, padx=2)
tk.Label(f_blur, text="カーネルサイズ:").pack(side=tk.LEFT, padx=(10, 2))
tk.Entry(f_blur, textvariable=blur_kernel_var, width=4).pack(side=tk.LEFT)
tk.Label(f_blur, text="(例: 確率0.5, サイズ3)").pack(side=tk.LEFT, padx=5)

# 7. 合成・生成画像の解像度
f_res = tk.Frame(frame_domain)
f_res.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_res, text="7. 生成画像解像度:", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_res, textvariable=target_res_var, width=6).pack(side=tk.LEFT)
tk.Label(f_res, text="px (正方形解像度, デフォルト: 512)").pack(side=tk.LEFT, padx=5)

# --- 📐 アノテーション (YOLOポリゴン化) パラメータ設定セクション (新規追加) ---
frame_anno = tk.LabelFrame(scrollable_frame, text="📐 YOLOポリゴン・アノテーション設定", fg="#1565c0", font=("sans-serif", 10, "bold"))
frame_anno.pack(fill=tk.X, padx=10, pady=5)

# 8. 微小ゴミ除去閾値
f_contour = tk.Frame(frame_anno)
f_contour.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_contour, text="8. 最小輪郭面積 (ゴミ除去):", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_contour, textvariable=min_contour_area_var, width=8).pack(side=tk.LEFT)
tk.Label(f_contour, text="px (デフォルト: 50.0 未満を除外)").pack(side=tk.LEFT, padx=5)

# 9. ポリゴン近似精度
f_eps = tk.Frame(frame_anno)
f_eps.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_eps, text="9. ポリゴン近似精度 (Epsilon):", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_eps, textvariable=poly_epsilon_var, width=8).pack(side=tk.LEFT)
tk.Label(f_eps, text="係数 (デフォルト: 0.002)").pack(side=tk.LEFT, padx=5)

# --- 📷 カメラ撮影画角 (ROI/ズーム) パラメータ設定セクション (新規追加) ---
frame_roi = tk.LabelFrame(scrollable_frame, text="📷 カメラ撮影画角 (ROI / ズーム) 設定", fg="#e65100", font=("sans-serif", 10, "bold"))
frame_roi.pack(fill=tk.X, padx=10, pady=5)

# 10. カメラ画角 (ROI/ズーム/クロップ)
f_roi1 = tk.Frame(frame_roi)
f_roi1.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_roi1, text="10. ズーム倍率 (ROI_ZOOM):", width=24, anchor="w").pack(side=tk.LEFT)
tk.Entry(f_roi1, textvariable=roi_zoom_var, width=8).pack(side=tk.LEFT)
tk.Label(f_roi1, text="倍 (デフォルト: 2.02)").pack(side=tk.LEFT, padx=5)

f_roi2 = tk.Frame(frame_roi)
f_roi2.pack(fill=tk.X, padx=5, pady=3)
tk.Label(f_roi2, text="クロップ割合 (Top/Bottom/Left/Right):", width=28, anchor="w").pack(side=tk.LEFT)
tk.Label(f_roi2, text="L:").pack(side=tk.LEFT)
tk.Entry(f_roi2, textvariable=roi_crop_left_var, width=5).pack(side=tk.LEFT, padx=2)
tk.Label(f_roi2, text="R:").pack(side=tk.LEFT)
tk.Entry(f_roi2, textvariable=roi_crop_right_var, width=5).pack(side=tk.LEFT, padx=2)
tk.Label(f_roi2, text="T:").pack(side=tk.LEFT)
tk.Entry(f_roi2, textvariable=roi_crop_top_var, width=5).pack(side=tk.LEFT, padx=2)
tk.Label(f_roi2, text="B:").pack(side=tk.LEFT)
tk.Entry(f_roi2, textvariable=roi_crop_bottom_var, width=5).pack(side=tk.LEFT, padx=2)

frame_domain_save = tk.Frame(scrollable_frame)
frame_domain_save.pack(fill=tk.X, padx=10, pady=5)
tk.Button(frame_domain_save, text="💾 ランダム化・アノテーション・ROI設定をデフォルト保存", command=save_all_domain_defaults, bg="#607D8B", fg="white", font=("sans-serif", 9, "bold")).pack(anchor="e", padx=5)

# --- AI学習パラメータ設定セクション ---
frame_train = tk.LabelFrame(scrollable_frame, text="YOLOv8 AI学習パラメータ")
frame_train.pack(fill=tk.X, padx=10, pady=5)

f_epochs = tk.Frame(frame_train)
f_epochs.pack(fill=tk.X, padx=5, pady=5)
tk.Label(f_epochs, text="学習周回数 (Epochs):", width=22, anchor="w").pack(side=tk.LEFT)
epochs_var = tk.StringVar(value="30")
tk.Entry(f_epochs, textvariable=epochs_var, width=8).pack(side=tk.LEFT)

f_batch = tk.Frame(frame_train)
f_batch.pack(fill=tk.X, padx=5, pady=5)
tk.Label(f_batch, text="同時処理数 (Batch):", width=22, anchor="w").pack(side=tk.LEFT)
batch_var = tk.StringVar(value="16")
tk.Entry(f_batch, textvariable=batch_var, width=8).pack(side=tk.LEFT)
tk.Label(f_batch, text="※GPUメモリ12GBなら16、CPUなら小さめ推奨", fg="gray").pack(side=tk.LEFT, padx=10)

# --- 実行ボタン (統合一括実行) ---
frame_btn = tk.Frame(scrollable_frame)
frame_btn.pack(fill=tk.X, padx=10, pady=10)
exec_btn = tk.Button(frame_btn, text="🚀 データ生成 ＆ YOLO学習 実行", command=start_pipeline, width=35, height=2, bg="#4CAF50", fg="white", font=("sans-serif", 10, "bold"))
exec_btn.pack()

# --- データ管理セクション ---
frame_manage = tk.LabelFrame(scrollable_frame, text="データ管理 (Dataset Management)")
frame_manage.pack(fill=tk.X, padx=10, pady=10)
btn_clear = tk.Button(frame_manage, text="🗑️ 現在のデータをクリア", command=clear_dataset, bg="#F44336", fg="white", font=("sans-serif", 9, "bold"), width=20)
btn_clear.pack(side=tk.LEFT, padx=5, pady=10)
btn_yaml = tk.Button(frame_manage, text="📝 data.yaml のみ更新", command=manual_generate_yaml, bg="#FFC107", font=("sans-serif", 9, "bold"), width=20)
btn_yaml.pack(side=tk.LEFT, padx=5, pady=10)
btn_save = tk.Button(frame_manage, text="📦 データセットをZIP保存", command=save_dataset_to_zip, bg="#2196F3", fg="white", font=("sans-serif", 9, "bold"), width=25)
btn_save.pack(side=tk.RIGHT, padx=5, pady=10)

root.mainloop()
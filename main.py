import os
import shutil
import time
from datetime import datetime
import numpy as np
import cv2
import mediapipe as mp
from plyer import filechooser

# --- Kivy Imports ---
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.behaviors import ButtonBehavior
from kivy.uix.slider import Slider
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.graphics import Color, Ellipse, Rectangle, Line, RoundedRectangle
from kivy.core.window import Window
from kivy.config import Config

# إعدادات Kivy
Config.set('input', 'mouse', 'mouse,multitouch_on_demand')
Config.set('graphics', 'multisamples', '0')
os.environ['KIVY_NO_ARGS'] = '1'

# --- TFLite Import (محاولة استيراد المكتبة المتاحة) ---
tflite = None
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        print("Warning: TensorFlow Lite not found. Gender detection will not work.")

# =========================================================================
# 🛠️ إعداد المسارات
# =========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FILTERS_BASE = os.path.join(BASE_DIR, 'filters')
MALE_DIR = os.path.join(FILTERS_BASE, 'male')
FEMALE_DIR = os.path.join(FILTERS_BASE, 'female')
MODEL_PATH = os.path.join(BASE_DIR, 'gender.tflite') # تأكدي أن الاسم صحيح

# إنشاء المجلدات إذا لم تكن موجودة
for d in [MALE_DIR, FEMALE_DIR]:
    if not os.path.exists(d):
        os.makedirs(d)

# =========================================================================
# 🧬 ثوابت ونقاط الوجه
# =========================================================================
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
INNER_LIPS = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# =========================================================================
# 🧠 كلاس تحديد الجنس (المعدل والذكي) ✅
# =========================================================================
class GenderDetector:
    def __init__(self, model_path):
        if tflite is None:
            raise ImportError("TFLite library not available")
        print(f"[INFO] Loading Gender Model from: {model_path}")
        self.interpreter = tflite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']
        print(f"[SUCCESS] Gender Model Loaded! Input shape: {self.input_shape}")

    def predict(self, face_img):
        try:
            if face_img is None or face_img.size == 0:
                print("[WARNING] Empty face image!")
                return "Unknown"
            
            # 1. تغيير الحجم
            target_h, target_w = self.input_shape[1], self.input_shape[2]
            resized = cv2.resize(face_img, (target_w, target_h))
            
            # 2. معالجة الألوان (Normalization)
            if self.input_details[0]['dtype'] == np.float32:
                # الطريقة الأكثر شيوعاً: القسمة على 255 لجعل القيم 0-1
                input_data = np.float32(resized) / 255.0
            else:
                input_data = np.array(resized, dtype=np.uint8)

            input_data = np.expand_dims(input_data, axis=0)
            self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
            self.interpreter.invoke()
            
            # 3. استلام النتيجة
            output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
            raw_out = output_data[0]
            
            # 4. تحليل النتيجة
            result = ""
            if len(raw_out) > 1:
                # إذا كان الموديل يخرج رقمين [Prob_Female, Prob_Male]
                result = "Male" if raw_out[1] > raw_out[0] else "Female"
                confidence = max(raw_out[0], raw_out[1]) * 100
            else:
                # إذا كان رقماً واحداً
                score = raw_out[0]
                result = "Male" if score > 0.5 else "Female"
                confidence = abs(score - 0.5) * 200  # تحويل للنسبة المئوية
            
            print(f"[RESULT] Gender Detected: {result} (Confidence: {confidence:.1f}%)")
            return result

        except Exception as e:
            print(f"[ERROR] Gender Predict Error: {e}")
            import traceback
            traceback.print_exc()
            return "Unknown"

# =========================================================================
# 🎨 فلاتر الألوان
# =========================================================================
class ColorFilters:
    @staticmethod
    def apply(frame, mode):
        try:
            if mode == 'Normal': return frame
            elif mode == 'Cinematic': 
                frame = cv2.applyColorMap(frame, cv2.COLORMAP_BONE)
                return cv2.addWeighted(frame, 0.4, frame, 0.6, 0)
            elif mode == 'Vivid': 
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hsv[:, :, 1] = cv2.multiply(hsv[:, :, 1], 1.5)
                return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            elif mode == 'Sepia':
                kernel = np.array([[0.272, 0.534, 0.131], [0.349, 0.686, 0.168], [0.393, 0.769, 0.189]])
                return cv2.transform(frame, kernel)
            elif mode == 'B&W':
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return frame
        except: return frame

# =========================================================================
# 🧠 محرك المعالجة (يتضمن قص الوجه)
# =========================================================================
class ImageProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.face_mesh_static = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.1)
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
        self.mp_selfie_seg = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_seg.SelfieSegmentation(model_selection=0)

    def get_landmarks_points(self, img, is_static=False):
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        processor = self.face_mesh_static if is_static else self.face_mesh
        results = processor.process(rgb)
        if not results.multi_face_landmarks: return None
        points = []
        for pt in results.multi_face_landmarks[0].landmark:
            points.append((int(pt.x * w), int(pt.y * h)))
        return np.array(points)

    def get_hand_mask(self, img):
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)
        hand_mask = np.zeros((h, w), dtype=np.uint8)
        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                points = np.array([(int(pt.x * w), int(pt.y * h)) for pt in hand_lms.landmark], dtype=np.int32)
                hull = cv2.convexHull(points)
                cv2.fillConvexPoly(hand_mask, hull, 255)
                hand_mask = cv2.dilate(hand_mask, np.ones((40,40), np.uint8), iterations=1)
                hand_mask = cv2.GaussianBlur(hand_mask, (51, 51), 0)
        return hand_mask

    # --- دالة قص الوجه (لإرساله للموديل) ---
    def get_face_crop(self, frame, landmarks):
        if landmarks is None or len(landmarks) == 0: return None
        h_frame, w_frame = frame.shape[:2]
        x_min, y_min = np.min(landmarks, axis=0)
        x_max, y_max = np.max(landmarks, axis=0)
        face_w = x_max - x_min
        face_h = y_max - y_min
        # إضافة هامش حول الوجه
        pad_x = int(face_w * 0.3)
        pad_y = int(face_h * 0.4) 
        crop_x1 = max(0, x_min - pad_x)
        crop_y1 = max(0, y_min - pad_y)
        crop_x2 = min(w_frame, x_max + pad_x)
        crop_y2 = min(h_frame, y_max + pad_y)
        if crop_x2 <= crop_x1 or crop_y2 <= crop_y1: return None
        return frame[crop_y1:crop_y2, crop_x1:crop_x2]

    def get_delaunay_triangles(self, points):
        rect = (0, 0, 5000, 5000)
        subdiv = cv2.Subdiv2D(rect)
        for p in points: subdiv.insert((float(p[0]), float(p[1])))
        triangle_list = subdiv.getTriangleList()
        delaunay_tri = []
        for t in triangle_list:
            pt_tri = [(t[0], t[1]), (t[2], t[3]), (t[4], t[5])]
            indices = []
            for pt in pt_tri:
                dists = np.linalg.norm(points - pt, axis=1)
                index = np.argmin(dists)
                if dists[index] < 1: indices.append(index)
            if len(indices) == 3: delaunay_tri.append(indices)
        return delaunay_tri

    def remove_bg(self, img):
        try:
            if img.shape[2] == 4: return img
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.segmentation.process(rgb)
            mask = (results.segmentation_mask * 255).astype(np.uint8)
            _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
            mask = cv2.GaussianBlur(mask, (3, 3), 0)
            b,g,r = cv2.split(img)
            return cv2.merge([b,g,r,mask])
        except: return img

    def warp_triangle(self, img1, img2, t1, t2):
        try:
            r1 = cv2.boundingRect(np.float32([t1]))
            r2 = cv2.boundingRect(np.float32([t2]))
            t1_rect = []; t2_rect = []; t2_rect_int = []
            for i in range(3):
                t1_rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
                t2_rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
                t2_rect_int.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
            img1_rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
            size = (r2[2], r2[3])
            warp_mat = cv2.getAffineTransform(np.float32(t1_rect), np.float32(t2_rect))
            img2_rect = cv2.warpAffine(img1_rect, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            mask = np.zeros((r2[3], r2[2], 4), dtype=np.float32)
            cv2.fillConvexPoly(mask, np.int32(t2_rect_int), (1.0, 1.0, 1.0, 1.0), 16, 0)
            img2_rect = img2_rect * mask
            y1, y2 = r2[1], r2[1]+r2[3]
            x1, x2 = r2[0], r2[0]+r2[2]
            if x1 < 0 or x2 > img2.shape[1] or y1 < 0 or y2 > img2.shape[0]: return
            img2[y1:y2, x1:x2] = img2[y1:y2, x1:x2] * (1 - mask) + img2_rect
        except: pass

    def apply_face_morph(self, frame, filter_data, landmarks, hand_mask, alpha):
        h, w = frame.shape[:2]
        warped_face = np.zeros((h, w, 4), dtype=np.uint8)
        src_img = filter_data['img']
        if src_img.shape[2] == 3: src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2BGRA)
        src_lms = filter_data['lms']
        triangles = filter_data['tris']
        for indices in triangles:
            t1 = [src_lms[indices[0]], src_lms[indices[1]], src_lms[indices[2]]]
            t2 = [landmarks[indices[0]], landmarks[indices[1]], landmarks[indices[2]]]
            self.warp_triangle(src_img, warped_face, t1, t2)
        alpha_channel = np.ascontiguousarray(warped_face[:, :, 3])
        cv2.fillConvexPoly(alpha_channel, landmarks[LEFT_EYE], 0)
        cv2.fillConvexPoly(alpha_channel, landmarks[RIGHT_EYE], 0)
        mouth_mask = np.zeros_like(alpha_channel)
        cv2.fillConvexPoly(mouth_mask, landmarks[INNER_LIPS], 255)
        mouth_mask = cv2.dilate(mouth_mask, np.ones((5,5), np.uint8), iterations=1)
        mouth_mask = cv2.GaussianBlur(mouth_mask, (15, 15), 0)
        alpha_int = alpha_channel.astype(np.int16)
        mouth_int = mouth_mask.astype(np.int16)
        result_alpha = np.clip(alpha_int - mouth_int, 0, 255).astype(np.uint8)
        alpha_channel = result_alpha
        if hand_mask is not None:
            hand_mask_inv = cv2.bitwise_not(hand_mask)
            alpha_channel = cv2.bitwise_and(alpha_channel, hand_mask_inv)
        alpha_channel = cv2.GaussianBlur(alpha_channel, (151, 151), 0)
        warped_face[:, :, 3] = alpha_channel
        filter_rgb = warped_face[:, :, :3].astype(float)
        filter_alpha = warped_face[:, :, 3].astype(float) / 255.0
        filter_alpha = filter_alpha * alpha
        filter_alpha_3ch = cv2.merge([filter_alpha, filter_alpha, filter_alpha])
        frame_float = frame.astype(float)
        result = filter_rgb * filter_alpha_3ch + frame_float * (1.0 - filter_alpha_3ch)
        return result.astype(np.uint8)

    def apply_smart_overlay(self, frame, img, landmarks, alpha, name=""):
        h, w = frame.shape[:2]
        h_src, w_src = img.shape[:2]
        src_pts = np.float32([[0, 0], [w_src, 0], [0, h_src], [w_src, h_src]])

        # تحويل الاسم لحروف صغيرة لتسهيل الفحص
        name_lower = name.lower()

        # ---------------------------------------------------------
        # الحالة 1: نظارات (Glasses)
        # ---------------------------------------------------------
        if "glass" in name_lower:
            # نركز على العيون (العين اليسرى 33، العين اليمنى 263)
            # ونأخذ نقطة بين الحواجب (168) كمرجع للارتفاع
            
            # عرض النظارة: المسافة بين طرفي الوجه عند العيون
            left_eye_outer = landmarks[33]
            right_eye_outer = landmarks[263]
            center_top = landmarks[168] # نقطة بين الحواجب
            
            # حساب العرض والمركز
            eye_dist = np.linalg.norm(left_eye_outer - right_eye_outer)
            
            # تكبير النظارة قليلاً لتكون أعرض من العيون
            scale = 2.0 
            width_new = eye_dist * scale
            height_new = width_new * (h_src / w_src) # الحفاظ على النسبة
            
            # مركز النظارة هو منتصف المسافة بين العينين
            center_x = (left_eye_outer[0] + right_eye_outer[0]) / 2
            center_y = (left_eye_outer[1] + right_eye_outer[1]) / 2
            
            # تحديد الزوايا الأربعة للنظارة حول المركز
            x1 = center_x - (width_new / 2)
            x2 = center_x + (width_new / 2)
            y1 = center_y - (height_new / 2)
            y2 = center_y + (height_new / 2)
            
            dst_pts = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])

        # ---------------------------------------------------------
        # الحالة 2: لحية (Beard)
        # ---------------------------------------------------------
        elif "beard" in name_lower:
            # نركز على النصف السفلي (الأنف 1، الذقن 152، الخدود 234 و 454)
            nose_tip = landmarks[1]
            chin = landmarks[152]
            left_cheek = landmarks[234]
            right_cheek = landmarks[454]
            
            # العرض: من الخد للخد
            face_width = np.linalg.norm(left_cheek - right_cheek)
            # الطول: من الأنف للذقن
            face_height = np.linalg.norm(nose_tip - chin)
            
            # ضبط الهوامش
            margin_x = face_width * 0.1 # زيادة 10% عرض
            margin_y_top = face_height * 0.3 # نرفع اللحية قليلاً فوق الأنف لتغطية الشارب
            margin_y_bottom = face_height * 0.2 # ننزلها تحت الذقن قليلاً
            
            x1 = left_cheek[0] - margin_x
            x2 = right_cheek[0] + margin_x
            y1 = nose_tip[1] - margin_y_top
            y2 = chin[1] + margin_y_bottom
            
            dst_pts = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])

        # ---------------------------------------------------------
        # الحالة 3: شارب (Mustache)
        # ---------------------------------------------------------
        elif "mustache" in name_lower or "moustache" in name_lower:
            # منطقة الشارب: فوق الشفة العليا
            nose_bottom = landmarks[2]  # أسفل الأنف
            upper_lip = landmarks[13]   # الشفة العليا
            left_mouth = landmarks[61]  # زاوية الفم اليسرى
            right_mouth = landmarks[291] # زاوية الفم اليمنى
            
            mouth_width = np.linalg.norm(left_mouth - right_mouth)
            
            # الشارب أعرض من الفم قليلاً
            margin_x = mouth_width * 0.2
            margin_y = mouth_width * 0.15
            
            x1 = left_mouth[0] - margin_x
            x2 = right_mouth[0] + margin_x
            y1 = nose_bottom[1] - margin_y
            y2 = upper_lip[1] + margin_y
            
            dst_pts = np.float32([[x1, y1], [x2, y1], [x1, y2], [x2, y2]])
            
        # ---------------------------------------------------------
        # الحالة 4: الافتراضي (شعر، قناع كامل، أو أي شيء آخر)
        # ---------------------------------------------------------
        else:
            # الكود القديم (الوجه كاملاً)
            top = landmarks[10]; bottom = landmarks[152]; left = landmarks[234]; right = landmarks[454]
            margin_x = np.linalg.norm(left - right) * 0.15 
            margin_y = np.linalg.norm(top - bottom) * 0.15
            dst_pts = np.float32([
                [left[0]-margin_x, top[1]-margin_y], [right[0]+margin_x, top[1]-margin_y],
                [left[0]-margin_x, bottom[1]+margin_y], [right[0]+margin_x, bottom[1]+margin_y]
            ])

        # ---------------------------------------------------------
        # التنفيذ (Warp & Blend)
        # ---------------------------------------------------------
        try:
            matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped_img = cv2.warpPerspective(img, matrix, (w, h))
            
            if warped_img.shape[2] == 4:
                alpha_mask = warped_img[:, :, 3] / 255.0
                
                # تقليل التنعيم للنظارات للحفاظ على حواف حادة
                if "glass" in name_lower:
                    alpha_mask = cv2.GaussianBlur(alpha_mask, (3, 3), 0)
                else:
                    alpha_mask = cv2.GaussianBlur(alpha_mask, (45, 45), 0)
                
                hand_mask = self.get_hand_mask(frame)
                if hand_mask is not None:
                    hand_mask_float = 1.0 - (hand_mask.astype(float) / 255.0)
                    alpha_mask = alpha_mask * hand_mask_float
                
                alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask]) * alpha
                fg = warped_img[:, :, :3].astype(float)
                bg = frame.astype(float)
                
                result = fg * alpha_mask + bg * (1.0 - alpha_mask)
                return result.astype(np.uint8)
            else: 
                return frame
        except:
            return frame

# =========================================================================
# 📱 عناصر الواجهة (UI)
# =========================================================================
class TextButton(Button):
    def __init__(self, **kwargs):
        self.btn_color = kwargs.pop('btn_color', (0.2, 0.2, 0.2, 0.8))
        super().__init__(**kwargs)
        self.background_color = (0,0,0,0)
        self.font_size = '16sp'
        self.bold = True
        self.markup = True
    def on_size(self, *args): self.draw()
    def on_pos(self, *args): self.draw()
    def draw(self, pressed=False):
        if not self.canvas: return
        self.canvas.before.clear()
        with self.canvas.before:
            c = self.btn_color
            if pressed: Color(c[0]+0.2, c[1]+0.2, c[2]+0.2, 1)
            else: Color(*c)
            RoundedRectangle(pos=self.pos, size=self.size, radius=[10])

class FilterSlotItem(ButtonBehavior, BoxLayout):
    def __init__(self, filter_id, image_texture=None, app_ref=None, **kwargs):
        self.text_content = kwargs.pop('text', '')
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (None, None)
        self.size = (80, 100)
        self.filter_id = filter_id
        self.app_ref = app_ref
        self.is_selected = False
        self.image_texture = image_texture
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.update_canvas()
    def update_canvas(self, *args):
        self.clear_widgets()
        icon_box = FloatLayout(size_hint=(1, 0.7))
        with icon_box.canvas.before:
            if self.is_selected:
                Color(1, 0.8, 0, 1) 
                Ellipse(pos=(self.x+2, self.y+22), size=(76, 76))
            Color(1, 1, 1, 1)
            if self.image_texture:
                Ellipse(pos=(self.x+5, self.y+25), size=(70, 70), texture=self.image_texture)
            else:
                Color(0.2, 0.2, 0.2, 0.5)
                Ellipse(pos=(self.x+5, self.y+25), size=(70, 70))
        if self.text_content:
            lbl = Label(text=self.text_content, font_size='30sp', bold=True, color=(1,0,0,1), pos_hint={'center_x':0.5, 'center_y':0.5})
            icon_box.add_widget(lbl)
        self.add_widget(icon_box)
        name_text = f"F{self.filter_id+1}" if self.filter_id >= 0 else "None"
        if self.filter_id == -1: name_text = "None"
        lbl_name = Label(text=name_text, font_size='12sp', size_hint=(1, 0.3), valign='top')
        self.add_widget(lbl_name)
    def on_press(self):
        if self.app_ref: self.app_ref.set_active_filter(self.filter_id)

class CustomCaptureButton(ButtonBehavior, BoxLayout):
    def __init__(self, label_text, bg_color, callback, **kwargs):
        super().__init__(**kwargs)
        self.size_hint = (None, None)
        self.size = (80, 80)
        self.bg_color = bg_color
        self.callback = callback
        self.bind(pos=self.update_canvas, size=self.update_canvas)
        self.add_widget(Label(text=label_text, font_size='14sp', bold=True))
        self.update_canvas()
    def update_canvas(self, *args):
        self.canvas.before.clear()
        with self.canvas.before:
            Color(*self.bg_color)
            Ellipse(pos=self.pos, size=self.size)
            Color(1, 1, 1, 1)
            Line(circle=(self.center_x, self.center_y, 41), width=2)
    def on_press(self):
        if self.callback: self.callback(self)

# =========================================================================
# 🚀 التطبيق الرئيسي
# =========================================================================

class SnapCamApp(App):
    def build(self):
        Window.clearcolor = (0,0,0,1)
        self.root = FloatLayout()
        self.processor = ImageProcessor()
        self.cam_id = 0
        self.capture = cv2.VideoCapture(self.cam_id)
        self.capture.set(3, 1280); self.capture.set(4, 720)
        self.mask_library = []
        self.current_mask_idx = -1
        self.curr_color = 'Normal'
        self.color_idx = 0
        self.color_modes = ['Normal', 'Cinematic', 'Vivid', 'Sepia', 'B&W']
        self.flash = False
        self.is_rec = False
        self.writer = None
        self.frame_bgr = None
        self.filter_buttons = []
        self.is_mirror = True
        self.image = Image(fit_mode="cover", size_hint=(1,1))
        self.root.add_widget(self.image)
        
        # تهيئة تحديد الجنس
        self.gender_detector = None
        self.detected_gender = "Female"  # نبدأ بـ Female عشان نشوف حاجة فوراً
        self.current_loaded_gender = None
        self.last_gender_check_time = 0
        
        print("=" * 60)
        print("[INFO] Initializing Gender Detection...")
        
        # محاولة تحميل الموديل
        self.gender_detector = None
        self.use_ai_detection = False
        
        if os.path.exists(MODEL_PATH):
            try:
                print("[INFO] Trying to load AI model...")
                self.gender_detector = GenderDetector(MODEL_PATH)
                self.use_ai_detection = True
                print("[SUCCESS] AI Model loaded! Using AI detection.")
                self.detected_gender = "Female"  # البداية
            except Exception as e:
                print(f"[WARNING] AI Model failed to load: {e}")
                print("[INFO] Using simple heuristic detection instead")
                self.use_ai_detection = False
                self.detected_gender = "Female"
        else:
            print(f"[WARNING] Model not found at: {MODEL_PATH}")
            print("[INFO] Using simple heuristic detection instead")
            self.use_ai_detection = False
            self.detected_gender = "Female"
        
        print("=" * 60)

        # UI Construction
        top_bar = BoxLayout(size_hint=(1, 0.1), pos_hint={'top':1}, padding=10, spacing=15)
        self.btn_flash = TextButton(text="Flash", size_hint=(None, 1), width=80)
        self.btn_flash.bind(on_press=self.toggle_flash)
        self.btn_color = TextButton(text="Color FX", size_hint=(None, 1), width=90)
        self.btn_color.bind(on_press=self.cycle_color)
        
        # زرار اختيار الجنس يدوياً
        self.btn_gender = TextButton(text="♀ Female", size_hint=(None, 1), width=100, btn_color=(0.5, 0, 0.5, 0.8))
        self.btn_gender.bind(on_press=self.toggle_gender_manual)
        
        self.btn_flip = TextButton(text="Flip Cam", size_hint=(None, 1), width=90)
        self.btn_flip.bind(on_press=self.flip_camera)
        self.btn_upload = TextButton(text="Upload +", size_hint=(None, 1), width=90, btn_color=(0.2, 0.6, 1, 0.8))
        self.btn_upload.bind(on_press=self.open_file_manager)
        top_bar.add_widget(self.btn_flash)
        top_bar.add_widget(self.btn_color)
        top_bar.add_widget(self.btn_gender)
        top_bar.add_widget(Label())
        top_bar.add_widget(self.btn_upload)
        top_bar.add_widget(self.btn_flip)
        self.root.add_widget(top_bar)
        
        self.lbl_info = Label(text="Initializing...", font_size='20sp', bold=True, pos_hint={'center_x':0.5, 'top':0.88})
        self.root.add_widget(self.lbl_info)
        
        self.slider = Slider(min=0, max=100, value=90, pos_hint={'center_x':0.5, 'y':0.20}, size_hint=(0.8, 0.05))
        self.root.add_widget(self.slider)
        
        actions_box = BoxLayout(size_hint=(None, None), size=(200, 90), spacing=30, pos_hint={'center_x':0.5, 'y':0.28})
        self.btn_snap = CustomCaptureButton("PHOTO", (1,1,1,0.2), self.take_snapshot)
        self.btn_rec = CustomCaptureButton("VIDEO", (1,0,0,0.2), self.toggle_recording)
        actions_box.add_widget(self.btn_snap)
        actions_box.add_widget(self.btn_rec)
        self.root.add_widget(actions_box)
        
        bottom_area = FloatLayout(size_hint=(1, 0.20), pos_hint={'bottom':1})
        with bottom_area.canvas.before:
            Color(0, 0, 0, 0.4)
            Rectangle(pos=bottom_area.pos, size=bottom_area.size)
        scroll = ScrollView(size_hint=(1, 1), pos_hint={'center_y':0.5}, do_scroll_x=True, do_scroll_y=False)
        self.grid = GridLayout(rows=1, size_hint_x=None, spacing=10, padding=10)
        self.grid.bind(minimum_width=self.grid.setter('width'))
        scroll.add_widget(self.grid)
        bottom_area.add_widget(scroll)
        self.root.add_widget(bottom_area)
        
        # تحميل مبدئي
        Clock.schedule_once(lambda dt: self.refresh_library_dynamic('Female'), 0.5)
        Clock.schedule_interval(self.update, 1.0/30.0)
        return self.root

    def refresh_library_dynamic(self, gender_type=None):
        target_dir = MALE_DIR if gender_type == 'Male' else FEMALE_DIR
        print(f"Loading filters from: {target_dir}")
        
        self.mask_library.clear()
        self.grid.clear_widgets()
        self.filter_buttons = []
        
        no_filt = FilterSlotItem(-1, text="X", app_ref=self)
        self.grid.add_widget(no_filt)
        self.filter_buttons.append(no_filt)
        
        files = []
        if os.path.exists(target_dir):
            for file in os.listdir(target_dir):
                if any(file.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.webp']):
                    files.append(os.path.join(target_dir, file))
        
        files.sort(key=os.path.getmtime)
        for fpath in files: self.process_and_add_filter(fpath)
        
        self.current_mask_idx = -1
        self.set_active_filter(-1)

    def process_and_add_filter(self, fpath):
        try:
            stream = open(fpath, "rb")
            bytes = bytearray(stream.read())
            numpyarray = np.asarray(bytes, dtype=np.uint8)
            img = cv2.imdecode(numpyarray, cv2.IMREAD_UNCHANGED)
            stream.close()
            if img is not None:
                if len(img.shape) == 2: img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
                elif img.shape[2] == 3: img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                img = self.processor.remove_bg(img)
                h, w = img.shape[:2]
                if w > 600: img = cv2.resize(img, (int(w*(600/w)), int(h*(600/w))))
                lms = self.processor.get_landmarks_points(img, is_static=True)
                mask_data = {"img": img,"type": "morph" if lms is not None else "overlay","lms": lms,"tris": self.processor.get_delaunay_triangles(lms) if lms is not None else None}
                thumb = cv2.resize(img, (75, 75))
                if thumb.shape[2] == 3: thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2BGRA)
                buf = cv2.flip(thumb, 0).tobytes()
                tex = Texture.create(size=(thumb.shape[1], thumb.shape[0]), colorfmt='bgra')
                tex.blit_buffer(buf, colorfmt='bgra', bufferfmt='ubyte')
                self.mask_library.append({"thumb": thumb, "data": mask_data, "name": os.path.basename(fpath)})
                btn = FilterSlotItem(len(self.mask_library)-1, image_texture=tex, app_ref=self)
                self.grid.add_widget(btn)
                self.filter_buttons.append(btn)
        except Exception as e: print(f"Error loading {fpath}: {e}")

    def open_file_manager(self, instance):
        filechooser.open_file(on_selection=self.handle_upload, multiple=True)

    def handle_upload(self, selection):
        if selection:
            target_dir = MALE_DIR if self.current_loaded_gender == 'Male' else FEMALE_DIR
            for fpath in selection:
                try:
                    filename = os.path.basename(fpath)
                    new_filename = f"{int(time.time_ns())}_{filename}"
                    new_path = os.path.join(target_dir, new_filename)
                    shutil.copy2(fpath, new_path)
                except: pass
            self.refresh_library_dynamic(self.current_loaded_gender)

    def set_active_filter(self, idx):
        self.current_mask_idx = idx
        name = "No Mask"
        if idx >= 0 and idx < len(self.mask_library):
            name = self.mask_library[idx]["name"]
        self.lbl_info.text = f"Gen: {self.detected_gender} | {name}"
        for btn in self.filter_buttons:
            btn.is_selected = (btn.filter_id == idx)
            btn.update_canvas()

    def update(self, dt):
        ret, frame = self.capture.read()
        if ret:
            if self.is_mirror: frame = cv2.flip(frame, 1)
            lms_cam = self.processor.get_landmarks_points(frame, is_static=False)
            
            # --- رسم مؤشر الكشف على الشاشة ---
            detection_status = "❌ No Face"
            detection_color = (0, 0, 255)  # أحمر
            
            if lms_cam is not None:
                detection_status = "✓ Face Detected"
                detection_color = (0, 255, 0)  # أخضر
                
                # رسم مستطيل حول الوجه المكتشف
                x_min, y_min = np.min(lms_cam, axis=0)
                x_max, y_max = np.max(lms_cam, axis=0)
                
                # إضافة padding للمستطيل
                padding = 20
                x_min = max(0, x_min - padding)
                y_min = max(0, y_min - padding)
                x_max = min(frame.shape[1], x_max + padding)
                y_max = min(frame.shape[0], y_max + padding)
                
                # رسم المستطيل
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), detection_color, 3)
                
                # عرض نص الكشف على المستطيل
                cv2.putText(frame, "FACE DETECTED", (x_min, y_min - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, detection_color, 2)
            
            # --- منطق كشف الجنس (كل 0.5 ثانية بدلاً من 1.5) ---
            current_time = time.time()
            if lms_cam is not None and (current_time - self.last_gender_check_time > 0.5):
                self.last_gender_check_time = current_time
                
                if self.use_ai_detection and self.gender_detector:
                    # استخدام AI Model
                    print("[INFO] Attempting AI gender detection...")
                    cropped_face = self.processor.get_face_crop(frame, lms_cam)
                    
                    if cropped_face is not None and cropped_face.size > 0:
                        print(f"[INFO] Face cropped successfully! Size: {cropped_face.shape}")
                        gender_res = self.gender_detector.predict(cropped_face)
                        if gender_res != "Unknown":
                            self.detected_gender = gender_res
                            print(f"[RESULT] AI Detection: {self.detected_gender}")
                            
                            if self.detected_gender != self.current_loaded_gender:
                                self.current_loaded_gender = self.detected_gender
                                print(f"[INFO] Switching filters to: {self.detected_gender}")
                                Clock.schedule_once(lambda dt: self.refresh_library_dynamic(self.detected_gender), 0)
                        else:
                            print("[WARNING] AI Detection returned Unknown")
                    else:
                        print("[ERROR] Failed to crop face!")
                
                else:
                    # استخدام Heuristic Detection (كشف بسيط بدون AI)
                    # قياس نسبة الوجه والفك للتخمين
                    try:
                        # حساب عرض وطول الوجه
                        x_coords = lms_cam[:, 0]
                        y_coords = lms_cam[:, 1]
                        
                        face_width = np.max(x_coords) - np.min(x_coords)
                        face_height = np.max(y_coords) - np.min(y_coords)
                        face_ratio = face_width / face_height if face_height > 0 else 1.0
                        
                        # حساب عرض الفك السفلي (jaw width)
                        jaw_left = lms_cam[234]   # نقطة الفك الأيسر
                        jaw_right = lms_cam[454]  # نقطة الفك الأيمن
                        jaw_width = np.linalg.norm(jaw_left - jaw_right)
                        
                        # حساب عرض الخدود (cheek width)
                        cheek_left = lms_cam[127]
                        cheek_right = lms_cam[356]
                        cheek_width = np.linalg.norm(cheek_left - cheek_right)
                        
                        # نسبة الفك للخدود
                        jaw_to_cheek_ratio = jaw_width / cheek_width if cheek_width > 0 else 1.0
                        
                        # قواعد محسّنة للكشف:
                        # - الوجوه الذكورية عادة أعرض وفكها أكبر
                        # - الوجوه الأنثوية أضيق وأكثر بيضاوية
                        
                        # افتراضياً: Female (لأن معظم المستخدمين إناث)
                        detected = "Female"
                        
                        # فقط لو الوجه عريض جداً → Male
                        if face_ratio > 0.85 and jaw_to_cheek_ratio > 0.92:
                            detected = "Male"
                        
                        # تحديث فقط لو اختلف
                        if detected != self.detected_gender:
                            self.detected_gender = detected
                            print(f"[RESULT] Heuristic Detection: {self.detected_gender} (face_ratio: {face_ratio:.2f}, jaw_ratio: {jaw_to_cheek_ratio:.2f})")
                            
                            if self.detected_gender != self.current_loaded_gender:
                                self.current_loaded_gender = self.detected_gender
                                Clock.schedule_once(lambda dt: self.refresh_library_dynamic(self.detected_gender), 0)
                    
                    except Exception as e:
                        print(f"[WARNING] Heuristic detection failed: {e}")

            # تطبيق الفلاتر
            hand_mask = self.processor.get_hand_mask(frame)
            if self.current_mask_idx >= 0 and self.current_mask_idx < len(self.mask_library):
                mask_item = self.mask_library[self.current_mask_idx]
                data = mask_item["data"]
                filter_name = mask_item["name"]
                
                if lms_cam is not None and data:
                    if data["type"] == "morph":
                        frame = self.processor.apply_face_morph(frame, data, lms_cam, hand_mask, self.slider.value/100.0)
                    else:
                        frame = self.processor.apply_smart_overlay(frame, data["img"], lms_cam, self.slider.value/100.0, name=filter_name)
            
            # --- عرض الجنس بشكل واضح جداً في النص ---
            h, w = frame.shape[:2]
            
            # اختيار اللون حسب الجنس
            if self.detected_gender == "Male":
                gender_color = (255, 255, 0)  # سيان (أزرق فاتح)
                gender_bg = (128, 64, 0)  # خلفية زرقاء داكنة
                gender_text = "MALE"
            elif self.detected_gender == "Female":
                gender_color = (255, 0, 255)  # ماجنتا (وردي فاقع)
                gender_bg = (128, 0, 128)  # خلفية وردية داكنة
                gender_text = "FEMALE"
            else:
                gender_color = (255, 255, 255)  # أبيض
                gender_bg = (80, 80, 80)  # رمادي
                gender_text = self.detected_gender.upper()
            
            # رسم صندوق كبير في وسط الأعلى
            box_width = 300
            box_height = 100
            box_x = (w - box_width) // 2  # في الوسط
            box_y = 80  # من فوق
            
            # رسم خلفية شبه شفافة
            overlay = frame.copy()
            cv2.rectangle(overlay, (box_x, box_y), (box_x + box_width, box_y + box_height), gender_bg, -1)
            cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
            
            # رسم إطار ملون
            cv2.rectangle(frame, (box_x, box_y), (box_x + box_width, box_y + box_height), gender_color, 4)
            
            # عرض الجنس بخط كبير جداً
            text_size = cv2.getTextSize(gender_text, cv2.FONT_HERSHEY_DUPLEX, 2.0, 4)[0]
            text_x = box_x + (box_width - text_size[0]) // 2
            text_y = box_y + (box_height + text_size[1]) // 2
            
            # رسم النص مع outline
            cv2.putText(frame, gender_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 0), 6)  # outline
            cv2.putText(frame, gender_text, (text_x, text_y), 
                       cv2.FONT_HERSHEY_DUPLEX, 2.0, gender_color, 4)  # النص نفسه
            
            # عرض حالة الكشف في أعلى الشاشة
            status_text = f"{detection_status}"
            cv2.putText(frame, status_text, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
            cv2.putText(frame, status_text, (10, 35), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, detection_color, 2)
            
            frame = ColorFilters.apply(frame, self.curr_color)
            if self.flash: frame = cv2.convertScaleAbs(frame, 1, 50)
            self.frame_bgr = frame
            
            filter_name = self.mask_library[self.current_mask_idx]["name"] if self.current_mask_idx >= 0 else "None"
            self.lbl_info.text = f"Gen: {self.detected_gender} | {filter_name} | {self.curr_color}"
            
            if self.is_rec and self.writer: self.writer.write(frame)
            
            buf = cv2.flip(frame, 0).tobytes()
            tex = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            tex.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            if self.image: self.image.texture = tex

    def toggle_gender_manual(self, i):
        """تبديل الجنس يدوياً بين Male و Female"""
        if self.detected_gender == "Female":
            self.detected_gender = "Male"
            i.text = "♂ Male"
            i.btn_color = (0, 0.5, 0.5, 0.8)  # أزرق
        else:
            self.detected_gender = "Female"
            i.text = "♀ Female"
            i.btn_color = (0.5, 0, 0.5, 0.8)  # وردي
        i.draw()
        # تحميل الفلاتر المناسبة
        if self.detected_gender != self.current_loaded_gender:
            self.current_loaded_gender = self.detected_gender
            self.refresh_library_dynamic(self.detected_gender)
    
    def toggle_flash(self, i):
        self.flash = not self.flash
        i.btn_color = (1,1,0,0.8) if self.flash else (0.2,0.2,0.2,0.8)
        i.draw()
    def cycle_color(self, i):
        self.color_idx = (self.color_idx + 1) % len(self.color_modes)
        self.curr_color = self.color_modes[self.color_idx]
        i.btn_color = (np.random.random(), np.random.random(), np.random.random(), 0.8)
        i.draw()
    def flip_camera(self, i):
        self.capture.release()
        self.cam_id = 1 - self.cam_id
        self.capture = cv2.VideoCapture(self.cam_id)
        self.capture.set(3, 1280); self.capture.set(4, 720)
    def take_snapshot(self, i):
        if self.frame_bgr is not None:
            name = f"SNAP_{datetime.now().strftime('%H%M%S')}.jpg"
            cv2.imwrite(name, self.frame_bgr)
            self.lbl_info.text = "Snapshot Saved!"
            i.bg_color = (1,1,1,1); i.update_canvas()
            Clock.schedule_once(lambda d: setattr(i, 'bg_color', (1,1,1,0.2)) or i.update_canvas(), 0.1)
    def toggle_recording(self, i):
        if not self.is_rec:
            if self.frame_bgr is not None:
                h, w = self.frame_bgr.shape[:2]
                name = f"REC_{datetime.now().strftime('%H%M%S')}.mp4"
                self.writer = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
                self.is_rec = True
                self.lbl_info.text = "Recording..."
                i.bg_color = (1,0,0,0.8); i.update_canvas()
        else:
            self.is_rec = False
            if self.writer: self.writer.release()
            self.lbl_info.text = "Video Saved!"
            i.bg_color = (1,0,0,0.2); i.update_canvas()
    def on_stop(self):
        self.capture.release()
        if self.writer: self.writer.release()

if __name__ == '__main__':
    SnapCamApp().run()
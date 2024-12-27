import os
import vlc
import threading
from tkinter import Tk, filedialog
from ttkbootstrap import Style
from ttkbootstrap.widgets import Button, Label
import cv2
import mediapipe as mp
import time

class FingerDetectionApp:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.is_running = True
        self.detected_fingers = 0

    def count_fingers(self, hand_landmarks):
        finger_tips = [4, 8, 12, 16, 20]
        landmarks = hand_landmarks.landmark

        thumb_tip = landmarks[finger_tips[0]]
        thumb_ip = landmarks[finger_tips[0] - 2]
        thumb_is_up = abs(thumb_tip.x - thumb_ip.x) > abs(thumb_tip.y - thumb_ip.y)

        finger_count = sum(
            landmarks[tip].y < landmarks[tip - 2].y for tip in finger_tips[1:]
        )

        if thumb_is_up:
            finger_count += 1

        return finger_count

    def start_detection(self, autoplay_bot):
        with mp.solutions.hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8) as hands:
            while self.cap.isOpened() and self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = hands.process(rgb_frame)

                if result.multi_hand_landmarks:
                    for hand_landmarks in result.multi_hand_landmarks:
                        finger_count = self.count_fingers(hand_landmarks)
                        self.detected_fingers = finger_count

                        # Control video player based on finger count
                        if finger_count == 5:
                            autoplay_bot.pause_video()
                        elif finger_count == 1:
                            autoplay_bot.Resume_video()

                        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)

                cv2.imshow("Hand Gesture Detection", frame)
                if cv2.waitKey(10) & 0xFF == 27:
                    self.is_running = False

        self.cap.release()
        cv2.destroyAllWindows()


class AutoplayBot:
    def __init__(self, root):
        self.root = root
        self.style = Style(theme="litera")
        self.media_player = None
        self.video_list = []
        self.current_index = 0
        self.playing = False

        self.gesture_control = FingerDetectionApp()
        self.detection_thread = threading.Thread(target=self.gesture_control.start_detection, args=(self,), daemon=True)
        self.detection_thread.start()

        self.setup_ui()

    def setup_ui(self):
        self.root.title("Autoplay Bot")
        self.root.geometry("600x400")

        Button(self.root, text="Select Folder", command=self.select_folder).pack(pady=10)
        Button(self.root, text="Play", command=self.play_video).pack(pady=10)
        Button(self.root, text="Pause", command=self.pause_video).pack(pady=10)
        Button(self.root, text="Resume", command=self.Resume_video).pack(pady=10)
        Button(self.root, text="Stop", command=self.stop_video).pack(pady=10)
        Button(self.root, text="Next", command=self.next_video).pack(pady=10)

        self.label = Label(self.root, text="No video playing.", anchor="center")
        self.label.pack(fill="x", pady=20)

    def select_folder(self):
        folder_path = filedialog.askdirectory()
        if folder_path:
            self.video_list = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith((".mp4", ".avi", ".mkv"))]
            self.current_index = 0
            if self.video_list:
                self.label.config(text="Folder selected. Ready to play.")

    def play_video(self):
        if self.video_list and not self.playing:
            if self.media_player:
                self.media_player.stop()
            video_path = self.video_list[self.current_index]
            self.media_player = vlc.MediaPlayer(video_path)
            self.media_player.play()
            self.playing = True
            self.label.config(text=f"Playing: {os.path.basename(video_path)}")
            time.sleep(1)

    def pause_video(self):
        if self.media_player and self.playing:
            self.media_player.pause()
            self.playing = False
            self.label.config(text="Paused.")
    
    def Resume_video(self):
        if self.media_player and not self.playing:
            self.media_player.play()
            self.playing = True
            self.label.config(text="resume.")

    def stop_video(self):
        if self.media_player:
            self.media_player.stop()
            self.playing = False
            self.label.config(text="Stopped.")

    def next_video(self):
        if self.video_list:
            self.current_index = (self.current_index + 1) % len(self.video_list)
            self.play_video()

    def on_close(self):
        self.gesture_control.is_running = False
        self.root.destroy()

if __name__ == "__main__":
    root = Tk()
    app = AutoplayBot(root)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()

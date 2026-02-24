import pygame
import webbrowser
import threading

class Alarm:
    def __init__(self, sound_file=None, video_url=None, use_sound=True, use_video=True):
        self.sound_file = sound_file
        self.video_url = video_url
        self.use_sound = use_sound
        self.use_video = use_video
        self.sound_playing = False
        self.video_opened = False  # Évite d'ouvrir plusieurs fois la même vidéo
        if self.use_sound and self.sound_file:
            pygame.mixer.init()
            pygame.mixer.music.load(self.sound_file)

    def trigger(self):
        if self.use_sound and self.sound_file and not self.sound_playing:
            pygame.mixer.music.play(-1)
            self.sound_playing = True
        if self.use_video and self.video_url and not self.video_opened:
            threading.Thread(target=lambda: webbrowser.open(self.video_url)).start()
            self.video_opened = True

    def stop(self):
        if self.use_sound and self.sound_playing:
            pygame.mixer.music.stop()
            self.sound_playing = False
        # Ne pas réinitialiser video_opened pour éviter de rouvrir après un arrêt/redémarrage

    def cleanup(self):
        if self.use_sound:
            pygame.mixer.quit()
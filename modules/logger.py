import logging
import os

class Logger:
    def __init__(self, log_file=None, level=logging.INFO):
        self.logger = logging.getLogger('DrowsinessDetector')
        self.logger.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def info(self, msg):
        self.logger.info(msg)

    def warning(self, msg):
        self.logger.warning(msg)

    def error(self, msg):
        self.logger.error(msg)
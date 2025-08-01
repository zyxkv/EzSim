import inspect
import os
import threading
import time

import numpy as np
import taichi as ti
from PIL import Image

import ezsim


def animate(imgs, filename=None, fps=60, preset="ultrafast"):
    """
    Create a video from a list of images.

    Args:
        imgs (list): List of input images.
        filename (str, optional): Name of the output video file. If not provided, the name will be default to the name of the caller file, with a timestamp and '.mp4' extension.
    """
    assert isinstance(imgs, list)
    if len(imgs) == 0:
        ezsim.logger.warning("No image to save.")
        return

    if filename is None:
        caller_file = inspect.stack()[-1].filename
        # caller file + timestamp + .mp4
        filename = os.path.splitext(os.path.basename(caller_file))[0] + f'_{time.strftime("%Y%m%d_%H%M%S")}.mp4'
    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)

    ezsim.logger.info(f'Saving video to ~<"{filename}">~...')
    from moviepy import ImageSequenceClip

    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(
        filename,
        fps=fps,
        logger=None,
        codec="libx264",
        preset=preset,
        # ffmpeg_params=["-crf", "0"],
    )
    ezsim.logger.info("Video saved.")


def save_img_arr(arr, filename="img.png"):
    assert isinstance(arr, np.ndarray)
    os.makedirs(os.path.abspath(os.path.dirname(filename)), exist_ok=True)
    img = Image.fromarray(arr)
    img.save(filename)
    ezsim.logger.info(f"Image saved to ~<{filename}>~.")


class Timer:
    def __init__(self, skip=False, level=0, ti_sync=False):
        self.accu_log = dict()
        self.skip = skip
        self.level = level
        self.ti_sync = ti_sync
        self.msg_width = 0
        self.reset()

    def reset(self):
        self.just_reset = True
        if self.level == 0 and not self.skip:
            print("─" * os.get_terminal_size()[0])
        if self.ti_sync and not self.skip:
            ti.sync()
        self.prev_time = self.init_time = time.perf_counter()

    def _stamp(self, msg="", _ratio=1.0):
        if self.skip:
            return

        if self.ti_sync:
            ti.sync()

        self.cur_time = time.perf_counter()
        self.msg_width = max(self.msg_width, len(msg))
        step_time = 1000 * (self.cur_time - self.prev_time) * _ratio
        accu_time = 1000 * (self.cur_time - self.init_time) * _ratio

        if msg not in self.accu_log:
            self.accu_log[msg] = [1, step_time, accu_time]
        else:
            self.accu_log[msg][0] += 1
            self.accu_log[msg][1] += step_time
            self.accu_log[msg][2] += accu_time

        if self.level > 0:
            prefix = " │  " * (self.level - 1)
            if self.just_reset:
                prefix += " ╭──"
            else:
                prefix += " ├──"
        else:
            prefix = ""

        print(
            f"{prefix}[{msg.ljust(self.msg_width)}] step: {step_time:5.3f}ms | accu: {accu_time:5.3f}ms | step_avg: {self.accu_log[msg][1]/self.accu_log[msg][0]:5.3f}ms | accu_avg: {self.accu_log[msg][2]/self.accu_log[msg][0]:5.3f}ms"
        )

        self.prev_time = time.perf_counter()
        self.just_reset = False

    def stamp(self, msg="", _ratio=1.0):
        return
        if self.skip:
            return

        if self.ti_sync:
            ti.sync()

        self.cur_time = time.perf_counter()
        self.msg_width = max(self.msg_width, len(msg))
        step_time = 1000 * (self.cur_time - self.prev_time) * _ratio
        accu_time = 1000 * (self.cur_time - self.init_time) * _ratio

        if msg not in self.accu_log:
            self.accu_log[msg] = [1, step_time, accu_time]
        else:
            self.accu_log[msg][0] += 1
            self.accu_log[msg][1] += step_time
            self.accu_log[msg][2] += accu_time

        if self.level > 0:
            prefix = " │  " * (self.level - 1)
            if self.just_reset:
                prefix += " ╭──"
            else:
                prefix += " ├──"
        else:
            prefix = ""

        print(
            f"{prefix}[{msg.ljust(self.msg_width)}] step: {step_time:5.3f}ms | accu: {accu_time:5.3f}ms | step_avg: {self.accu_log[msg][1]/self.accu_log[msg][0]:5.3f}ms | accu_avg: {self.accu_log[msg][2]/self.accu_log[msg][0]:5.3f}ms"
        )

        self.prev_time = time.perf_counter()
        self.just_reset = False


timers = dict()


def create_timer(name=None, new=False, level=0, ti_sync=False, skip_first_call=False):
    if name is None:
        return Timer()
    else:
        if name in timers and not new:
            timer = timers[name]
            timer.skip = False
            timer.reset()
            return timer
        else:
            timer = Timer(skip=skip_first_call, level=level, ti_sync=ti_sync)
            timers[name] = timer
            return timer


class Rate:
    def __init__(self, rate):
        self.rate = rate
        self.last_time = time.perf_counter()

    def sleep(self):
        current_time = time.perf_counter()
        sleep_duration = 1.0 / self.rate - (current_time - self.last_time)
        if sleep_duration > 0:
            time.sleep(sleep_duration)
        self.last_time = time.perf_counter()


class FPSTracker:
    def __init__(self, n_envs, alpha=0.95):
        self.last_time = None
        self.n_envs = n_envs
        self.dt_ema = None
        self.alpha = alpha

    def step(self):
        current_time = time.perf_counter()

        if self.last_time:
            dt = current_time - self.last_time
        else:
            self.last_time = current_time
            return

        if self.dt_ema:
            self.dt_ema = self.alpha * self.dt_ema + (1 - self.alpha) * dt
        else:
            self.dt_ema = dt
        fps = 1 / self.dt_ema
        
        if self.n_envs > 0:
            self.total_fps = fps * self.n_envs
            ezsim.logger.info(
                f"Running at ~<{self.total_fps:,.2f}>~ FPS (~<{fps:.2f}>~ FPS per env, ~<{self.n_envs}>~ envs)."
            )
        else:
            self.total_fps = fps
            ezsim.logger.info(f"Running at ~<{fps:.2f}>~ FPS.")
        self.last_time = current_time

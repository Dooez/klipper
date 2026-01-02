# Calibration of heater PID settings
#
# Copyright (C) 2016-2018  Kevin O'Connor <kevin@koconnor.net>
#
# This file may be distributed under the terms of the GNU GPLv3 license.
import logging
import math

import numpy as np
import scipy as sp
from numpy.matlib import float64

from . import heaters
from .heaters import ControlHeater, Heater


class PIDCalibrate:
    def __init__(self, config):
        self.printer = config.get_printer()
        gcode = self.printer.lookup_object("gcode")
        gcode.register_command(
            "PID_CALIBRATE", self.cmd_PID_CALIBRATE, desc=self.cmd_PID_CALIBRATE_help
        )

    cmd_PID_CALIBRATE_help = "Run PID calibration test"

    def cmd_PID_CALIBRATE(self, gcmd):
        heater_name = gcmd.get("HEATER")
        target = gcmd.get_float("TARGET")
        write_file = gcmd.get_int("WRITE_FILE", 0)
        pheaters = self.printer.lookup_object("heaters")
        try:
            heater = pheaters.lookup_heater(heater_name)
        except self.printer.config_error as e:
            raise gcmd.error(str(e))
        self.printer.lookup_object("toolhead").get_last_move_time()
        # calibrate = ControlAutoTune(heater, target)
        calibrate = ControlZiNi(heater, target)
        old_control = heater.set_control(calibrate)
        try:
            pheaters.set_temperature(heater, target, True)
        except self.printer.command_error as e:
            heater.set_control(old_control)
            raise
        heater.set_control(old_control)
        if write_file:
            calibrate.write_file("/tmp/heattest.txt")
        if calibrate.check_busy(0.0, 0.0, 0.0):
            raise gcmd.error("pid_calibrate interrupted")
        # Log and report results
        Kp, Ki, Kd = calibrate.calc_final_pid()
        logging.info("Autotune: final: Kp=%f Ki=%f Kd=%f", Kp, Ki, Kd)
        gcmd.respond_info(
            "PID parameters: pid_Kp=%.3f pid_Ki=%.3f pid_Kd=%.3f\n"
            "The SAVE_CONFIG command will update the printer config file\n"
            "with these parameters and restart the printer." % (Kp, Ki, Kd)
        )
        # Store results for SAVE_CONFIG
        cfgname = heater.get_name()
        configfile = self.printer.lookup_object("configfile")
        configfile.set(cfgname, "control", "pid")
        configfile.set(cfgname, "pid_Kp", "%.3f" % (Kp,))
        configfile.set(cfgname, "pid_Ki", "%.3f" % (Ki,))
        configfile.set(cfgname, "pid_Kd", "%.3f" % (Kd,))


class ControlZiNi(ControlHeater):
    class Limits:
        temp_delta: float = 2.0  # deg C
        start_kp: float = 25.0
        max_kp: float = 100.0
        iteration_duration: float = 10  # sec
        Ku_cutoff_freq: float = 2 / iteration_duration  # Hz
        oscillation_peak_ratio: float = 100
        max_iteration_count: int = 100

    def __init__(self, heater: Heater, calibrate_temp: float):
        self.heater: Heater = heater
        self.heater_max_power: float = heater.get_max_power()
        self.calibrate_temp: float = calibrate_temp

        # Heating control
        self.iteration: bool = False
        self.limits: ControlZiNi.Limits = self.Limits()
        self.limits.max_kp = self.heater_max_power / self.limits.temp_delta

        # Ultimate gain search
        self.current_Kp: float = self.limits.max_kp
        self.peak_oscillation_ratio: float = 0
        self.peak_oscillation_period: float = 0
        self.found_peak: bool = False
        self.rollback_count: int = 0

        self.pwm_samples: list[float] = []
        self.time_samples: list[float] = []
        self.temp_samples: list[float] = []

        self.history_pwm_samples: list[list[float]] = []
        self.history_time_samples: list[list[float]] = []
        self.history_spec: list[np.typing.NDArray[float64]] = []

        self.iteration_count: int = 0

        self.found_Kp: bool = False

    # Iteration with current Kp in search of ultimate gain Ku
    def iterate(self, read_time: float, temp: float, target_temp: float):
        if not self.iteration:
            return
        if read_time > self.time_samples[0] + self.limits.iteration_duration:
            self.iteration = False
            self.set_pwm(read_time, 0)
            self.check_Ku()
            return

        self.time_samples.append(read_time)
        self.temp_samples.append(temp)

        temp_err = target_temp - temp
        pwr = self.current_Kp * temp_err
        bounded_pwr = max(0.0, min(self.heater_max_power, pwr))
        self.set_pwm(read_time, bounded_pwr)

    # Heater control
    def set_pwm(self, read_time, value):
        self.pwm_samples.append(value)
        self.time_samples.append(read_time)
        self.heater.set_pwm(read_time, value)

    def reset_iteration(self):
        self.pwm_samples = []
        self.time_samples = []
        self.temp_samples = []

    def temperature_update(self, read_time: float, temp: float, target_temp: float):
        if self.found_Kp:
            return

        if not self.iteration:
            self.heater.alter_target(self.calibrate_temp)
            if temp >= target_temp - self.limits.temp_delta:
                self.reset_iteration()
                self.iteration = True
            else:
                self.heater.set_pwm(read_time, 99999999999)

        self.iterate(read_time, temp, target_temp)

    def check_busy(self, eventtime, smoothed_temp, target_temp):
        return not self.found_Kp

    # Analysis
    def check_Ku(self):
        time_samples = [t - self.time_samples[1] for t in self.time_samples]
        totalTime = time_samples[-1]
        avgTd = totalTime / len(time_samples)
        xp = np.array(time_samples)
        fp = np.array(self.pwm_samples)
        x = np.linspace(0, avgTd * (len(xp) - 1))
        f = np.interp(x, xp, fp)
        spec = sp.fft.fft(f)
        spec = spec[: len(spec) // 2]
        freqStep = 1 / totalTime
        nCutoff = math.ceil(self.limits.Ku_cutoff_freq / freqStep)
        spec[0:nCutoff] = 0

        self.history_pwm_samples.append(self.pwm_samples)
        self.history_time_samples.append(time_samples)
        self.history_pwm_samples.append(spec)

        pwr_spec = np.square(spec)
        idx = np.argmax(pwr_spec)
        peak_ratio = pwr_spec[idx] / (
            np.sum(pwr_spec[0 : idx - 1]) + np.sum(pwr_spec[idx + 1 : -1])
        )

        small_Kp_step = 1
        big_Kp_step = 10

        if not self.found_peak and peak_ratio > self.peak_oscillation_ratio:
            self.peak_oscillation_ratio = peak_ratio
            self.peak_oscillation_period = 1 / (freqStep * idx)
            if self.current_Kp >= self.limits.max_kp:
                # TODO: warning
                self.found_Kp = True
                return
            self.current_Kp += big_Kp_step
            return

        if not self.found_peak:
            self.found_peak = True
            self.rollback_count = big_Kp_step * 2 // small_Kp_step
            self.current_Kp -= big_Kp_step

        if self.rollback_count == 0:
            self.found_Kp = True
            return
        self.current_Kp += small_Kp_step
        self.rollback_count -= 1

    def calc_final_pid(self):
        Ku = self.current_Kp
        Tu = self.peak_oscillation_period
        Ti = 0.5 * Tu
        Kp = 0.6 * Ku
        Td = 0.125 * Tu
        Ki = Kp / Ti
        Kd = Kp * Td
        logging.info(
            "Autotune: Ku=%f Tu=%f  Kp=%f Ki=%f Kd=%f",
            self.heater_max_power,
            Ku,
            Tu,
            Kp,
            Ki,
            Kd,
        )
        return Kp, Ki, Kd

    # Offline analysis helper
    def write_file(self, filename):
        pwm = [
            "pwm: %.3f %.3f" % (time, value)
            for time, value in zip(self.time_samples, self.pwm_samples)
        ]
        out = [
            "%.3f %.3f" % (time, temp)
            for time, temp in zip(self.time_samples, self.temp_samples)
        ]
        f = open(filename, "w")
        f.write("\n".join(pwm + out))
        f.close()


TUNE_PID_DELTA = 5.0


class ControlAutoTune:
    def __init__(self, heater, target):
        self.heater = heater
        self.heater_max_power = heater.get_max_power()
        self.calibrate_temp = target
        # Heating control
        self.heating = False
        self.peak = 0.0
        self.peak_time = 0.0
        # Peak recording
        self.peaks = []
        # Sample recording
        self.last_pwm = 0.0
        self.pwm_samples = []
        self.temp_samples = []

    # Heater control
    def set_pwm(self, read_time, value):
        if value != self.last_pwm:
            self.pwm_samples.append((read_time + self.heater.get_pwm_delay(), value))
            self.last_pwm = value
        self.heater.set_pwm(read_time, value)

    def temperature_update(self, read_time, temp, target_temp):
        self.temp_samples.append((read_time, temp))
        # Check if the temperature has crossed the target and
        # enable/disable the heater if so.
        if self.heating and temp >= target_temp:
            self.heating = False
            self.check_peaks()
            self.heater.alter_target(self.calibrate_temp - TUNE_PID_DELTA)
        elif not self.heating and temp <= target_temp:
            self.heating = True
            self.check_peaks()
            self.heater.alter_target(self.calibrate_temp)
        # Check if this temperature is a peak and record it if so
        if self.heating:
            self.set_pwm(read_time, self.heater_max_power)
            if temp < self.peak:
                self.peak = temp
                self.peak_time = read_time
        else:
            self.set_pwm(read_time, 0.0)
            if temp > self.peak:
                self.peak = temp
                self.peak_time = read_time

    def check_busy(self, eventtime, smoothed_temp, target_temp):
        if self.heating or len(self.peaks) < 12:
            return True
        return False

    # Analysis
    def check_peaks(self):
        self.peaks.append((self.peak, self.peak_time))
        if self.heating:
            self.peak = 9999999.0
        else:
            self.peak = -9999999.0
        if len(self.peaks) < 4:
            return
        self.calc_pid(len(self.peaks) - 1)

    def calc_pid(self, pos):
        temp_diff = self.peaks[pos][0] - self.peaks[pos - 1][0]
        time_diff = self.peaks[pos][1] - self.peaks[pos - 2][1]
        # Use Astrom-Hagglund method to estimate Ku and Tu
        amplitude = 0.5 * abs(temp_diff)
        Ku = 4.0 * self.heater_max_power / (math.pi * amplitude)
        Tu = time_diff
        # Use Ziegler-Nichols method to generate PID parameters
        Ti = 0.5 * Tu
        Td = 0.125 * Tu
        Kp = 0.6 * Ku * heaters.PID_PARAM_BASE
        Ki = Kp / Ti
        Kd = Kp * Td
        logging.info(
            "Autotune: raw=%f/%f Ku=%f Tu=%f  Kp=%f Ki=%f Kd=%f",
            temp_diff,
            self.heater_max_power,
            Ku,
            Tu,
            Kp,
            Ki,
            Kd,
        )
        return Kp, Ki, Kd

    def calc_final_pid(self):
        cycle_times = [
            (self.peaks[pos][1] - self.peaks[pos - 2][1], pos)
            for pos in range(4, len(self.peaks))
        ]
        midpoint_pos = sorted(cycle_times)[len(cycle_times) // 2][1]
        return self.calc_pid(midpoint_pos)

    # Offline analysis helper
    def write_file(self, filename):
        pwm = ["pwm: %.3f %.3f" % (time, value) for time, value in self.pwm_samples]
        out = ["%.3f %.3f" % (time, temp) for time, temp in self.temp_samples]
        f = open(filename, "w")
        f.write("\n".join(pwm + out))
        f.close()


def load_config(config):
    return PIDCalibrate(config)

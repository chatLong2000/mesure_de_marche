import time

class Flasher:
    COMMAND_TIMEOUT = 0.0
    # COMMAND_TIMEOUT = 1

    _trig_off = -1
    _trig_expo = -1
    _trig_shift = -1
    _flash_on = -1
    _flash_off = -1

    def __init__(self, ser):
        self.ser = ser

    def _send_cmd(self, cmd: str) -> None:
        try:
            for c in cmd:
                self.ser.write(c.encode())

            if not cmd.endswith(';'):
                self.ser.write(b';')

            time.sleep(self.COMMAND_TIMEOUT)
        except Exception as e:
            print(f"Error sending command: {e}")

    def on(self) -> None:
        self._send_cmd("trig.en:1;")

    def off(self) -> None:
        self._send_cmd("trig.en:0;")

    def trig_off(self, val: int) -> None:
        self._trig_off = val
        self._send_cmd(f"trig.off:{val};")

    def trig_expo(self, val: int) -> None:
        self._trig_expo = val
        self._send_cmd(f"trig.expo:{val};")

    def trig_shift(self, val: int) -> None:
        self._trig_shift = val
        self._send_cmd(f"trig.shift:{val};")

    def flash_on(self, val: int) -> None:
        self._flash_on = val
        self._send_cmd(f"flash.on:{val};")

    def flash_off(self, val: int) -> None:
        self._flash_off = val
        self._send_cmd(f"flash.off:{val};")

    @property
    def current_trig_off(self) -> int:
        return self._trig_off

    @property
    def current_trig_expo(self) -> int:
        return self._trig_expo

    @property
    def current_trig_shift(self) -> int:
        return self._trig_shift

    @property
    def current_flash_on(self) -> int:
        return self._flash_on

    @property
    def current_flash_off(self) -> int:
        return self._flash_off

    @property
    def flash_frequency_hz(self) -> float:
        if self._trig_off <= 0:
            return 0.0
        return 1e6 / self._trig_off

    def print_config(self) -> None:
        print(f"trig_off = {self._trig_off}")
        print(f"trig_expo = {self._trig_expo}")
        print(f"trig_shift = {self._trig_shift}")
        print(f"flash_on = {self._flash_on}")
        print(f"flash_off = {self._flash_off}")

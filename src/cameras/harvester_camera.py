from harvesters.core import Harvester

class HarvesterCamera:
    """Pilotage caméra GigE via Harvester (GenICam)."""

    def __init__(self, cti_path, exposure_us=10000, gain=0.0):
        self.cti_path = cti_path
        self.exposure_us = exposure_us
        self.gain = gain

        self.harvester = None
        self.ia = None
        self.connected = False
        self.hw_trigger = False

        self.width = 0
        self.height = 0

    def connect(self, device_index=0):
        self.harvester = Harvester()
        self.harvester.add_file(self.cti_path)
        self.harvester.update()

        print(self.harvester.device_info_list)

        if not self.harvester.device_info_list:
            raise RuntimeError("No camera found")

        # Create image acquirer
        self.ia = self.harvester.create(device_index)

        # IMPORTANT: use remote_device for camera parameters
        node_map = self.ia.remote_device.node_map

        # Exposure
        if hasattr(node_map, "ExposureAuto"):
            node_map.ExposureAuto.value = "Off"
        # if hasattr(node_map, "ExposureTime"):
        #     node_map.ExposureTime.value = self.exposure_us
        if hasattr(node_map, "ExposureMode"):
            node_map.ExposureMode.value = "TriggerWidth"
            # node_map.ExposureMode.value = "Timed"

        # Trigger configuration
        try:
            if hasattr(node_map, "TriggerSelector"):
                node_map.TriggerSelector.value = "FrameStart"

            node_map.TriggerMode.value = "On"
            node_map.TriggerSource.value = "Line1"
            node_map.TriggerActivation.value = "RisingEdge"

            self.hw_trigger = True
            print("[CAM] Hardware trigger enabled")

        except Exception:
            if hasattr(node_map, "TriggerMode"):
                node_map.TriggerMode.value = "Off"

            self.hw_trigger = False
            print("[CAM] Free-run mode")

        # Resolution
        if hasattr(node_map, "Width"):
            self.width = node_map.Width.value

        if hasattr(node_map, "Height"):
            self.height = node_map.Height.value

        self.connected = True
        return True

    def start_acquisition(self):
        self.ia.start()

    def stop_acquisition(self):
        self.ia.stop()

    def capture_frame(self, timeout_ms=5000):
        try:
            with self.ia.fetch(timeout=timeout_ms) as buffer:
                if not buffer.payload.components:
                    return None

                component = buffer.payload.components[0]
                frame = component.data.reshape(
                    component.height,
                    component.width
                )

                return frame.copy()

        except Exception:
            return None

    def set_exposure(self, exposure_us):
        self.exposure_us = exposure_us
        try:
            self.ia.device.node_map.ExposureTime.value = exposure_us
        except Exception:
            pass

    def disconnect(self):
        if self.ia:
            try:
                self.ia.destroy()
            except:
                pass
        if self.harvester:
            self.harvester.reset()

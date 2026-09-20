class CalibrationStatus:

    def __init__(self, total_frames=60):

        self.total_frames = total_frames
        self.current_frame = 0
        self.completed = False

    def update(self, current_frame):

        self.current_frame = min(
            current_frame,
            self.total_frames
        )

        if self.current_frame >= self.total_frames:
            self.completed = True

        progress = (
            self.current_frame /
            self.total_frames
        ) * 100

        if self.completed:
            status = "COMPLETE"
        else:
            status = "CALIBRATING"

        return {
            "progress": progress,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "completed": self.completed,
            "status": status
        }

    def get_result(self):

        progress = (
            self.current_frame /
            self.total_frames
        ) * 100

        return {
            "progress": progress,
            "current_frame": self.current_frame,
            "total_frames": self.total_frames,
            "completed": self.completed,
            "status": (
                "COMPLETE"
                if self.completed
                else "CALIBRATING"
            )
        }
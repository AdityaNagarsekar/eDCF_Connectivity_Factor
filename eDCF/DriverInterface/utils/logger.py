import io
import sys
import json

class LoggerCapture(io.StringIO):
    """
    A custom output stream that:
      - Captures every print statement into a list (log_list).
      - Also sends the text to the *original* sys.stdout so it still appears
        in the PyCharm (or wherever the normal Python console is).
    """
    def __init__(self):
        super().__init__()
        self.original_stdout = sys.stdout  # Keep reference to what PyCharm uses
        self.log_list = []

    def write(self, txt):
        """
        Every time something is printed:
        1) Append it to log_list for later retrieval.
        2) Send it to the original_stdout so PyCharm console prints it.
        3) Also store it in our own StringIO buffer by calling super().write(txt).
        """
        self.log_list.append(txt)
        self.original_stdout.write(txt)
        self.original_stdout.flush()
        super().write(txt)

    def get_logs_as_json(self):
        """
        Return the captured logs as a JSON string, for example:
        {"logs": ["line 1\n", "line 2\n", ...]}
        """
        return json.dumps({"logs": self.log_list})

    def clear_logs(self):
        """
        Clear our in-memory list and the StringIO buffer.
        """
        self.log_list.clear()
        self.seek(0)
        self.truncate(0)


# Instantiate a global capture instance
global_log_capture = LoggerCapture()

# Redirect sys.stdout so that *all* print statements in Python code
# will go through our LoggerCapture.
sys.stdout = global_log_capture


def get_logs_as_json():
    """
    Return the logs from our global capture instance as JSON text.
    """
    return global_log_capture.get_logs_as_json()


def clear_logs():
    """
    Clear the captured logs.
    """
    global_log_capture.clear_logs()

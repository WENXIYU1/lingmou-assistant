"""Hidden widget smoke test; no camera, mouse injection, or user config writes."""
from pathlib import Path
import tempfile
from unittest.mock import patch
from apps.desktop.ui import PracticeApp


def main():
    with tempfile.TemporaryDirectory() as directory:
        app = PracticeApp(Path(directory) / 'settings.json')
        app.withdraw()
        app.update_idletasks()
        # Explicit frame time: synchronous smoke calls can share one Windows
        # monotonic clock tick; the real safety core must reject duplicates.
        now = 100.0
        def frame():
            nonlocal now
            now += .05
            with patch('apps.desktop.ui.time.monotonic', return_value=now):
                app.tick()
        app.restore_signal()
        frame()
        app.resume()
        assert app.controller.can_practice
        app.lose_signal()
        app.restore_signal()
        frame()
        assert app.controller.paused
        app.resume()
        assert app.controller.can_practice
        app.pause()
        assert app.controller.paused
        assert not app.adaptation.system_control_allowed
        app.left = app.right = True
        frame()
        assert app.actions == 0
        app.release_eyes()
        app.shutdown()
    print('UI smoke passed (hidden widgets only, not visual acceptance)')


if __name__ == '__main__':
    main()

"""Synthetic legacy-formula comparison; never imports unsafe legacy code."""
import json
from apps.desktop.config import Settings
from apps.desktop.interaction import Controller
from apps.desktop.vision import Observation


def main():
    # Legacy formula reproduces static-review behavior; these are not camera data.
    settings = Settings(smoothing_seconds=0, deadzone_x=.0035, deadzone_y=.0035)
    controller = Controller(settings)
    controller.observe(Observation(0))
    controller.resume()
    controller.observe(Observation(.05))
    x, y = controller.observe(Observation(.10, .52, .5001))[0]
    legacy_x = .5 * .8 + (.5 + .02 * 14) * .2
    legacy_y = .5 * .8 + (.5 + .0001 * 14) * .2
    print(json.dumps({
        'input': 'synthetic normalized feature delta (0.02, 0.0001)',
        'legacy_output_delta': [round(legacy_x - .5, 6), round(legacy_y - .5, 6)],
        'new_output_delta': [round(x - .5, 6), round(y - .5, 6)],
        'interpretation': 'Only validates axis independence and configured gain; NOT real eye accuracy.',
    }, indent=2))


if __name__ == '__main__':
    main()

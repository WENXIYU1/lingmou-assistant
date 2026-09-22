"""Visible geometry regression, synthetic UI only; no camera access."""
from apps.desktop.camera_ui import CameraApp


def main():
    app=CameraApp()
    try:
        app.update()
        app.message.set('短提示')
        app.update_idletasks()
        before=(app.canvas.winfo_width(),app.canvas.winfo_height())
        app.message.set('校准失败原因与详细说明。'*30)
        app.update_idletasks()
        after=(app.canvas.winfo_width(),app.canvas.winfo_height())
        assert before==after,(before,after)
        print('Status text preserves canvas geometry:',before,after)
    finally:
        app.destroy()


if __name__=='__main__':
    main()

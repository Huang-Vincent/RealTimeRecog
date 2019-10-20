from kivy.app import App
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.graphics import Line, Color
from kivy.core.window import Window

class DrawInput(Widget):

    def on_touch_down(self, touch):
        print(touch)
        if touch.button == 'left':
            with self.canvas:
                Color(1, 1, 1)
                touch.ud["line"] = Line(point=(touch.x, touch.y), width = 3)
        else:
            self.canvas.clear();

    def on_touch_move(self, touch):
        print(touch)
        if touch.button == 'left':
            touch.ud["line"].points += (touch.x, touch.y)

    def on_touch_up(self, touch):
        print(touch)
        if touch.button == 'left':
            self.export_to_png("pic.png")

class Application(App):
    def build(self):
        return DrawInput()

Config.set('input', 'mouse', 'mouse, multitouch_on_demand')
Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '280')
Config.set('graphics', 'height', '280')
if __name__ == "__main__":
    Application().run()

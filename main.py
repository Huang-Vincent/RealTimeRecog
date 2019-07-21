from kivy.app import App
from kivy.config import Config
from kivy.uix.widget import Widget
from kivy.graphics import Line


class DrawInput(Widget):
    def on_touch_down(self, touch):
        print(touch)
        with self.canvas:
            touch.ud["line"] = Line(point=(touch.x, touch.y))

    def on_touch_move(self, touch):
        print(touch)
        touch.ud["line"].points += (touch.x, touch.y)


class Application(App):
    def build(self):
        return DrawInput()


Config.set('graphics', 'resizable', False)
Config.set('graphics', 'width', '200')
Config.set('graphics', 'height', '200')
if __name__ == "__main__":
    Application().run()

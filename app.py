from kivy.app import App

from kivy.uix.label import Label

class SimpleKivy4(App):
    def build(self):
        return Label(text="Hello World!")

if __name__ == "__main__":
    SimpleKivy4().run()

from gi import require_version
require_version('Gtk', '4.0')
from gi.repository import Gtk, GLib, Gdk, GdkPixbuf, GObject

class GifPaintable(GObject.Object, Gdk.Paintable):
    def __init__(self, path):
        super().__init__()
        self.animation = GdkPixbuf.PixbufAnimation.new_from_file(path)
        self.iterator = self.animation.get_iter()
        self.delay = self.iterator.get_delay_time()
        self.timeout = GLib.timeout_add(self.delay, self.on_delay)

        self.invalidate_contents()

    def on_delay(self):
        delay = self.iterator.get_delay_time()
        self.timeout = GLib.timeout_add(delay, self.on_delay)
        self.invalidate_contents()

        return GLib.SOURCE_REMOVE

    def do_get_intrinsic_height(self):
        return self.animation.get_height()

    def do_get_intrinsic_width(self):
        return self.animation.get_width()

    def invalidate_contents(self):
        self.emit("invalidate-contents")

    def do_snapshot(self, snapshot, width, height):
        timeval = GLib.TimeVal()
        timeval.tv_usec = GLib.get_real_time()
        self.iterator.advance(timeval)
        pixbuf = self.iterator.get_pixbuf()
        texture = Gdk.Texture.new_for_pixbuf(pixbuf)

        texture.snapshot(snapshot, width, height)
